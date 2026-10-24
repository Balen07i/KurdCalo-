# reviewer.py
#
# Second-opinion multimodal quality-control reviewer for KurdCaloBot.
#
# ARCHITECTURE (see docs/decision_rules.md's own §9 for why the knowledge
# layer is split into separate files by change-reason - same logic
# applies here): this module is a COMPLETELY SEPARATE provider/pipeline
# from gemini_queue.py/vision.py. It does not share Gemini's API key,
# quota tracking, pacing, retry logic, or circuit breaker - a review-
# provider outage or rate limit can never affect Gemini's pipeline, and
# vice versa. Gemini remains the sole primary recognition/nutrition
# model; nothing here replaces it. This is a deliberate constraint, not
# an oversight - see the module-level rejected-alternatives note at the
# bottom of this file.
#
# WHY NOT THE google-genai SDK OR ANOTHER SDK: every realistic free
# multimodal option evaluated (Groq, Mistral, OpenRouter) exposes an
# OpenAI-compatible /chat/completions endpoint, so one small stdlib HTTP
# call covers all of them - REVIEWER_BASE_URL/REVIEWER_MODEL swap is
# enough to change provider, no new code path needed. Uses urllib
# (stdlib) rather than adding httpx/requests as a new dependency -
# keeps the dependency surface exactly as small as before this change,
# which matters for Railway build reliability (see the last audit's
# "Railway startup/import/path safety" section - fewer new dependencies
# is strictly fewer new ways for a deploy to break).
#
# SAFE-BY-DEFAULT: if REVIEWER_API_KEY is not set, every function below
# is a fast no-op and the bot behaves EXACTLY as it did before this
# module existed. This mirrors the existing GEMINI_MODEL_POOL pattern
# (unset = sensible default, not a crash).

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import urllib.error
import urllib.request

import knowledge

logger = logging.getLogger(__name__)

# --- Configuration (all overridable via env, all with safe defaults) ----

REVIEWER_API_KEY = os.environ.get("REVIEWER_API_KEY", "").strip()
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
REVIEWER_BASE_URL = os.environ.get("REVIEWER_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
REVIEWER_TIMEOUT_SECONDS = float(os.environ.get("REVIEWER_TIMEOUT_SECONDS", "10.0"))

# Gating thresholds (see should_review) - when a normally-skippable
# high-confidence result still gets a second look. All conservative
# defaults chosen to target the specific failure modes already observed
# (large-portion overconfidence, over-decomposed meals) without reviewing
# the common simple case.
REVIEWER_KCAL_THRESHOLD = int(os.environ.get("REVIEWER_KCAL_THRESHOLD", "1200"))
REVIEWER_ITEM_COUNT_THRESHOLD = int(os.environ.get("REVIEWER_ITEM_COUNT_THRESHOLD", "4"))
# Random sample of otherwise-skipped simple/high-confidence cases, purely
# to keep a future eval dataset fresh (see design doc §9 - the logging
# table itself is intentionally NOT part of this change; this knob is a
# no-op until that's built). Off by default - 0.0 changes nothing.
REVIEWER_SAMPLE_RATE = float(os.environ.get("REVIEWER_SAMPLE_RATE", "0.0"))

_CONFIDENCE_TIERS = ["نزم", "مامناوەند", "بەرز"]  # low -> medium -> high

_CLARIFICATION_NUDGE_KURDISH = (
    "ئەم هەڵسەنگاندنە هەندێک نادڵنیایی تێدایە - "
    "ئەگەر پێویستە وردتر بزانین، تکایە زانیاری زیاتر بنێرە."
)

_REVIEW_JSON_SCHEMA_HINT = """
Respond ONLY with a JSON object in exactly this shape:
{
  "review_performed": true,
  "food_checks": [
    {
      "original_name": "...",
      "identification_verdict": "confirmed" | "corrected" | "uncertain",
      "corrected_food_id": null,
      "reasoning": "short justification",
      "hallucination_flag": false,
      "portion_plausibility": "plausible" | "implausible_too_high" | "implausible_too_low" | "cannot_judge",
      "macro_plausibility": "plausible" | "implausible" | "cannot_judge"
    }
  ],
  "confidence_calibration": {
    "reviewer_assessment": "justified" | "overconfident" | "underconfident",
    "recommended_confidence": "بەرز" | "مامناوەند" | "نزم"
  },
  "kurdish_wording_issues": ["..."],
  "decision_rule_compliance": {
    "should_have_asked_clarification": false,
    "reasoning": ""
  },
  "final_recommendation": "accept_as_is" | "apply_corrections" | "flag_for_clarification" | "insufficient_info",
  "reviewer_confidence_in_this_review": "high" | "medium" | "low"
}
food_checks must have exactly one entry per food in the original result,
in the same order. corrected_food_id must be null unless
identification_verdict is "corrected", and must then be a real id from
the reference data provided - never invent one.
""".strip()


def is_enabled() -> bool:
    """False (fast, no import-time cost beyond reading one env var) means
    every other function in this module is a no-op. This is the single
    switch that determines whether the reviewer exists at all for this
    deployment."""
    return bool(REVIEWER_API_KEY)


def should_review(result: dict) -> bool:
    """Gating: review the minority of results where a second opinion is
    actually likely to earn its cost (latency + reviewer-provider quota).
    See the design doc's "gating" section for the reasoning behind each
    condition - each one maps directly to an observed failure mode.
    Never called if is_enabled() is False (bot.py checks that first)."""
    if result.get("status") != "ok":
        return False  # no_food/failed/rate_limited results have nothing to review

    confidence = result.get("confidence")
    foods = result.get("foods") or []

    if confidence == "نزم":
        return True
    if confidence == "مامناوەند":
        return True

    # confidence == "بەرز" (high) from here on - only review if something
    # specifically undercuts that self-reported confidence:
    if any(not f.get("matched_glossary") for f in foods):
        return True

    matched_ids = [f["food_id"] for f in foods if f.get("food_id")]
    if matched_ids:
        rules = knowledge.get_relevant_recognition_context(matched_ids)
        if any(p.get("severity") in ("medium", "high") for p in rules["confusion_pairs"]):
            return True

    if result.get("total_kcal", 0) > REVIEWER_KCAL_THRESHOLD:
        return True
    if len(foods) >= REVIEWER_ITEM_COUNT_THRESHOLD:
        return True

    if REVIEWER_SAMPLE_RATE > 0 and random.random() < REVIEWER_SAMPLE_RATE:
        return True

    return False


async def review_result(image_bytes: bytes, media_type: str, result: dict) -> dict | None:
    """Runs the second-opinion review. Returns the parsed review dict, or
    None on ANY failure (disabled, timeout, network error, malformed
    response) - None always means "review did not happen", and the
    caller (bot.py) must treat that identically to "skipped by gating":
    keep Gemini's original result completely unmodified. This function
    never raises."""
    if not is_enabled():
        return None

    try:
        return await asyncio.to_thread(_call_reviewer_sync, image_bytes, media_type, result)
    except Exception:
        logger.exception("[REVIEWER] Unexpected error - continuing with Gemini's original result")
        return None


def _call_reviewer_sync(image_bytes: bytes, media_type: str, result: dict) -> dict | None:
    foods = result.get("foods") or []
    food_ids = [f["food_id"] for f in foods if f.get("food_id")]

    # For foods Gemini itself claimed matched_glossary=True without our
    # code ever verifying it (see vision.py's _finalize_result - that
    # branch trusts Gemini's self-report and never calls
    # find_glossary_match), attempt an independent, READ-ONLY lookup here
    # purely so the reviewer has a food_id to fetch context for. This
    # does not touch or trust vision.py's result in any way - if it finds
    # nothing, that food is simply reviewed with name-only context, which
    # is a safe degradation, not an error.
    for f in foods:
        if not f.get("food_id"):
            m = knowledge.find_glossary_match(f.get("name_kurdish", ""))
            if m and m.get("id"):
                food_ids.append(m["id"])

    food_context = knowledge.get_food_context_for_review(food_ids)
    recognition_context = knowledge.get_relevant_recognition_context(food_ids)

    payload = _build_request_payload(image_bytes, media_type, result, food_context, recognition_context)

    try:
        req = urllib.request.Request(
            url=f"{REVIEWER_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {REVIEWER_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=REVIEWER_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("[REVIEWER] HTTP %s from reviewer API - skipping review: %.200s",
                        exc.code, exc.read().decode("utf-8", errors="replace"))
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("[REVIEWER] Network/timeout error - skipping review: %s", exc)
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("[REVIEWER] Malformed HTTP response - skipping review: %s", exc)
        return None

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("[REVIEWER] Unexpected response shape from reviewer API - skipping review: %s", exc)
        return None

    review = _extract_json(content)
    if review is None:
        logger.warning("[REVIEWER] Could not parse reviewer JSON output - skipping review")
        return None

    return _validate_and_default(review, expected_food_count=len(foods))


def _build_request_payload(image_bytes, media_type, result, food_context, recognition_context) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    gemini_summary = {
        "foods": [
            {
                "name_kurdish": f.get("name_kurdish"),
                "food_id": f.get("food_id"),
                "matched_glossary": f.get("matched_glossary"),
                "portion_kurdish": f.get("portion_kurdish"),
                "kcal": f.get("kcal"),
                "protein_g": f.get("protein_g"),
                "carbs_g": f.get("carbs_g"),
                "fat_g": f.get("fat_g"),
            }
            for f in (result.get("foods") or [])
        ],
        "total_kcal": result.get("total_kcal"),
        "confidence": result.get("confidence"),
    }

    user_text = (
        "Gemini's result to review:\n" + json.dumps(gemini_summary, ensure_ascii=False) +
        "\n\nRelevant food database entries (reference nutrition, NOT an "
        "unquestionable source - use judgment, especially against "
        "\"assumption\"-tagged entries):\n" + json.dumps(food_context, ensure_ascii=False) +
        "\n\nRelevant recognition rules for this result:\n" +
        json.dumps(recognition_context, ensure_ascii=False) +
        "\n\n" + _REVIEW_JSON_SCHEMA_HINT
    )

    return {
        "model": REVIEWER_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": knowledge.build_reviewer_instructions()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ],
            },
        ],
    }


def _extract_json(raw_text: str):
    """Tolerant JSON extraction - mirrors vision.py's _extract_json.
    Duplicated rather than imported: reviewer.py is deliberately
    independent of vision.py's internals, matching the "completely
    separate provider" requirement at the code level, not just the
    API-key level."""
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_and_default(review: dict, expected_food_count: int) -> dict:
    """Defensive defaults, same philosophy as vision.py's
    _finalize_result - a partially-malformed reviewer response degrades
    to safe defaults (effectively 'no opinion') rather than being
    discarded entirely or crashing the merge step."""
    review.setdefault("review_performed", True)
    review.setdefault("food_checks", [])
    review.setdefault("confidence_calibration", {})
    review["confidence_calibration"].setdefault("reviewer_assessment", "justified")
    review["confidence_calibration"].setdefault("recommended_confidence", None)
    review.setdefault("kurdish_wording_issues", [])
    review.setdefault("decision_rule_compliance", {})
    review["decision_rule_compliance"].setdefault("should_have_asked_clarification", False)
    review.setdefault("final_recommendation", "accept_as_is")
    review.setdefault("reviewer_confidence_in_this_review", "low")

    if len(review["food_checks"]) != expected_food_count:
        # Mismatched count means we cannot safely align corrections to
        # foods by position (see apply_review) - treat as if no
        # per-food corrections were proposed, but keep whatever
        # top-level confidence/clarification signal is present, since
        # those don't depend on positional alignment.
        logger.warning(
            "[REVIEWER] food_checks count (%d) != expected (%d) - "
            "per-food corrections disabled for this review, top-level "
            "signals still applied", len(review["food_checks"]), expected_food_count,
        )
        review["food_checks"] = []

    return review


# --- Merging a review into Gemini's result -------------------------------

def _infer_portion_grams(food: dict) -> float | None:
    """
    Infers the physical portion size (in grams) Gemini implicitly
    identified for this food item, so a correction can scale the
    corrected food's nutrition to the SAME portion rather than silently
    replacing it with the database's flat "one typical serving" figure
    (see the bug this fixes: a 400g identified portion was being
    overwritten with a ~100g reference value with no scaling at all).

    Gemini's structured output has no explicit grams field - only a
    Kurdish portion description string (portion_kurdish) and a kcal
    estimate. Parsing the Kurdish string for a number is NOT done here:
    it's free text ("٤٠٠ گرام", "یەک پلێت", "نادیار"/unclear are all
    valid values) and unreliably extracting a number from it would be
    exactly the kind of silent guess decision_rules.md §3e warns
    against. Instead, this backs the portion out from a number we
    already have and trust: Gemini's own kcal estimate, divided by the
    ORIGINALLY-matched food's per-100g density (grams is a property of
    the physical portion on the plate, not of which specific food it
    turns out to be - so the ORIGINAL identification's density is the
    right thing to divide by, even when the identification itself is
    what's being corrected).

    Returns None whenever this can't be computed reliably - the ONLY
    correct response to that, per the explicit requirement, is to NOT
    scale-and-overwrite; the caller must treat None as "cannot convert
    the portion" and fall back to the disagreement-signal path instead.
    """
    original_food_id = food.get("food_id")
    if not original_food_id:
        return None  # nothing to derive a density from - see docstring
    original_entry = knowledge.get_food(original_food_id)
    if not original_entry:
        return None
    per100 = original_entry.get("nutrition_per_100g") or {}
    kcal_per_100g = per100.get("kcal")
    gemini_kcal = food.get("kcal")
    if not isinstance(kcal_per_100g, (int, float)) or kcal_per_100g <= 0:
        return None
    if not isinstance(gemini_kcal, (int, float)) or gemini_kcal <= 0:
        return None

    grams = (gemini_kcal / kcal_per_100g) * 100.0

    # Sanity bound: reject an implausible implied portion (e.g. a
    # near-zero or multi-kilogram single-item result, which more likely
    # means the original kcal/density numbers were themselves unreliable
    # than that the portion is real) rather than propagating a nonsense
    # scale factor into numbers we're about to present as authoritative.
    if not (0 < grams <= 3000):
        return None
    return grams


def _scale_nutrition_to_grams(entry: dict, grams: float) -> dict | None:
    """Scales a database entry's nutrition_per_100g to the given portion
    size. Returns None if the entry doesn't have complete per-100g data
    to scale from - caller must treat that the same as an unconvertible
    portion (don't overwrite)."""
    per100 = entry.get("nutrition_per_100g") or {}
    keys = ("kcal", "protein_g", "carbs_g", "fat_g")
    if not all(isinstance(per100.get(k), (int, float)) for k in keys):
        return None
    factor = grams / 100.0
    return {k: per100[k] * factor for k in keys}


def apply_review(result: dict, review: dict | None) -> dict:
    """Conservative merge. review=None (disabled/skipped/failed) returns
    result completely unchanged - this is the overwhelmingly common case
    and must be a true no-op. When a review WAS performed, a numeric
    correction is only ever applied when ALL of the following hold:
      1. The proposed corrected_food_id is a real, grounded entry.
      2. That entry's own confidence tag is "verified" - an "estimated"
         or "assumption" entry is reference/context only and may never
         auto-overwrite Gemini's numbers, even with a valid ID and a high
         reviewer-confidence review (explicit requirement).
      3. The identified portion can be reliably converted to grams (see
         _infer_portion_grams) so the correction is scaled to the SAME
         portion Gemini identified, not silently replaced with the
         database's flat "one serving" figure.
      4. The reviewer reported high confidence in its own review.
    Any one of these failing means NO numbers are overwritten - it falls
    through to the disagreement-signal path (confidence capped down, or
    flagged for clarification) instead, per decision_rules.md's "if
    uncertain, preserve and flag rather than guess" principle."""
    if not review or not review.get("review_performed"):
        return result

    result["reviewer_reviewed"] = True
    applied_correction = False

    if (
        review.get("final_recommendation") == "apply_corrections"
        and review.get("reviewer_confidence_in_this_review") == "high"
    ):
        foods = result.get("foods") or []
        checks = review.get("food_checks") or []
        if len(checks) == len(foods):  # alignment guard - see _validate_and_default
            for food, check in zip(foods, checks):
                if check.get("identification_verdict") != "corrected":
                    continue
                fid = check.get("corrected_food_id")
                entry = knowledge.get_food(fid) if fid else None
                if not entry:
                    continue  # ungrounded proposal - ignored, not applied

                if entry.get("confidence") != "verified":
                    logger.info(
                        "[REVIEWER] Correction to %s proposed but its database "
                        "entry is tagged %r, not 'verified' - reference/context "
                        "only, not auto-applying numbers.", fid, entry.get("confidence"),
                    )
                    continue

                grams = _infer_portion_grams(food)
                if grams is None:
                    logger.info(
                        "[REVIEWER] Correction to %s proposed but the identified "
                        "portion could not be reliably converted to grams - not "
                        "auto-applying numbers (would silently replace the "
                        "identified portion with a flat serving size).", fid,
                    )
                    continue

                scaled = _scale_nutrition_to_grams(entry, grams)
                if scaled is None:
                    continue  # entry missing complete per-100g data - don't guess

                names = entry.get("names") or {}
                food["name_kurdish"] = names.get("kurdish", food.get("name_kurdish"))
                food["emoji"] = entry.get("emoji", food.get("emoji"))
                food["kcal"] = round(scaled["kcal"])
                food["protein_g"] = round(scaled["protein_g"])
                food["carbs_g"] = round(scaled["carbs_g"])
                food["fat_g"] = round(scaled["fat_g"])
                food["food_id"] = fid
                food["matched_glossary"] = True
                food["reviewer_corrected"] = True
                food["reviewer_corrected_portion_g"] = round(grams, 1)  # for
                # transparency/debugging - not displayed to the user, but
                # makes it possible to sanity-check a correction in logs
                applied_correction = True

        if applied_correction:
            result["foods"] = foods
            result["total_kcal"] = sum(f.get("kcal", 0) for f in foods)
            result["total_protein_g"] = sum(f.get("protein_g", 0) for f in foods)
            result["total_carbs_g"] = sum(f.get("carbs_g", 0) for f in foods)
            result["total_fat_g"] = sum(f.get("fat_g", 0) for f in foods)
            recommended = review.get("confidence_calibration", {}).get("recommended_confidence")
            if recommended in _CONFIDENCE_TIERS:
                result["confidence"] = recommended

    if not applied_correction:
        if review.get("confidence_calibration", {}).get("reviewer_assessment") == "overconfident":
            result["confidence"] = _downgrade_confidence(result.get("confidence"))

        needs_clarification = (
            review.get("decision_rule_compliance", {}).get("should_have_asked_clarification")
            or review.get("final_recommendation") in ("flag_for_clarification", "insufficient_info")
        )
        if needs_clarification:
            result["needs_clarification"] = True
            result["confidence"] = _downgrade_confidence(result.get("confidence"))
            note = (result.get("note_kurdish") or "").strip()
            result["note_kurdish"] = (note + " " + _CLARIFICATION_NUDGE_KURDISH).strip()

    if review.get("kurdish_wording_issues"):
        # Advisory only, per design - logged for later manual review, never
        # auto-rewritten (that would need another model call and adds
        # complexity beyond this integration's scope).
        logger.info("[REVIEWER] Kurdish wording issues flagged: %s", review["kurdish_wording_issues"])

    result["reviewer_applied_correction"] = applied_correction
    return result


def _downgrade_confidence(current: str | None) -> str:
    """One tier down, never up - بەرز->مامناوەند->نزم, floor at نزم.
    Unrecognized/missing input defaults to the safest (lowest) tier
    rather than guessing."""
    if current not in _CONFIDENCE_TIERS:
        return "نزم"
    idx = _CONFIDENCE_TIERS.index(current)
    return _CONFIDENCE_TIERS[max(0, idx - 1)]


# --- Rejected alternatives (recorded so this isn't re-litigated later) --
#
# - Running the reviewer INSIDE gemini_queue.py's worker loop: rejected -
#   would mix two providers' failure modes (pacing, circuit breaker,
#   quota) into one piece of state. Kept as an independent call from
#   bot.py instead (see design doc §2).
# - Letting the reviewer freely overwrite kcal/macros with its own
#   numbers: rejected - only a full database entry (via corrected_food_id)
#   is trusted; a bare suggested number from the reviewer model itself is
#   explicitly NOT grounded enough and is never applied (see design doc §7).
# - Auto-regenerating note_kurdish/insight_kurdish on a wording flag:
#   rejected for this round - would need an extra model call; wording
#   issues are logged only, numbers/confidence are unaffected by them.
# - Parsing portion_kurdish's free-text Kurdish string for a gram figure
#   (e.g. extracting "400" from "٤٠٠ گرام"): rejected - it's unstructured
#   text that can just as easily read "یەک پلێت" or "نادیار", and
#   unreliably parsing it would be exactly the kind of silent guess
#   decision_rules.md §3e warns against. Portion is instead backed out
#   from Gemini's own kcal number divided by the originally-matched
#   food's per-100g density (see _infer_portion_grams) - a number we
#   already have and trust, not a new parsing surface.
# - Applying a corrected identification's NAME while leaving Gemini's
#   original (now-mismatched) numbers in place when the entry isn't
#   "verified" or the portion can't be converted: rejected - a food
#   labeled "beef" showing macros computed under a "lamb" assumption is
#   arguably more misleading than not correcting at all. A correction is
#   all-or-nothing: name+numbers together, or neither.
