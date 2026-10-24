# gemini_queue.py
#
# REDESIGNED for maximum daily capacity on the free Gemini tier. Previous
# rounds optimized request-level efficiency (caching, batching) in
# addition to this round's daily-quota-focused work below. NOTE (verified
# during a later audit): an earlier edit here had claimed "duplicate/
# concurrent submissions are genuinely uncommon for this bot's usage
# pattern" as the reasoning for setting BATCH_SIZE=1, which fully disabled
# batching rather than just deprioritizing it. That reasoning didn't hold
# up under inspection - bot.py deliberately enables concurrent_updates(True)
# specifically so multiple users' photo submissions CAN be in flight at
# once (see that setting's own comment in bot.py), and submit_photo_job()
# below enqueues into the per-model queue without any serializing step
# first - so concurrent submissions landing in the same queue is a real,
# intended scenario, not a rare edge case. BATCH_SIZE has been restored
# to 3 accordingly (see the constant below). This round's redesign
# otherwise targets the actual lever that matters most: squeezing the
# most out of each model's DAILY quota, and using every model in the pool
# as fully as possible.
#
# Core architecture, in priority order of impact:
#
#   1. PER-MODEL DEDICATED QUEUES + WORKERS. Each model in the pool gets
#      its own queue and its own worker task, running truly concurrently.
#      A Flash-Lite request never waits behind a Flash request for no
#      reason - the old design funneled everything through one shared
#      sequential worker even though Flash and Flash-Lite are completely
#      independent quota pools on Google's side.
#
#   2. PROACTIVE DAILY BUDGET TRACKING. Every model has a configured daily
#      request budget (see vision.MODEL_POOL). We count requests locally
#      and stop routing to a model BEFORE it's exhausted, instead of
#      finding out reactively via a wave of 429s (each of which costs a
#      real request and up to ~90s of retry time for nothing).
#
#   3. BUDGET-PACED THROTTLING. Each model self-paces to spread its
#      remaining daily budget across the REMAINING hours of the day
#      (ideal_interval = seconds_left_in_day / budget_left). This is the
#      difference between serving users all day and going dark by 2pm
#      after burning the whole budget on an early morning rush.
#
#   4. SMART ROUTING ACROSS THE WHOLE POOL. At submission time, jobs go
#      to whichever healthy model currently has the most PROPORTIONAL
#      daily headroom - this naturally load-balances traffic across every
#      model in the pool over the course of the day, rather than
#      exhausting a "preferred" model first and only using others as a
#      reactive fallback.
#
#   5. Reactive circuit breaker (unchanged in spirit from before) remains
#      as the ground-truth safety net for when the configured daily
#      budget estimate turns out to be too generous.
#
#   6. Caching, dynamic cooldown, and batching are retained (real, if
#      modest, wins) but are no longer the focus - see the README for the
#      production-data reasoning on why they're capped at their current
#      contribution.
#
# bot.py should call submit_photo_job() instead of vision.estimate_calories()
# directly. Image optimization is the CALLER's responsibility (do it
# before submitting) so oversized images never sit in a queue.

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone

from vision import (
    DEFAULT_DAILY_BUDGET, DEFAULT_RPM_LIMIT, MODEL_POOL, MODEL_RPM_LIMIT,
    compute_dhash, estimate_calories, estimate_calories_batch, get_http_stats, hamming_distance,
)

try:
    from zoneinfo import ZoneInfo
    _QUOTA_RESET_TZ = ZoneInfo(os.environ.get("GEMINI_QUOTA_RESET_TIMEZONE", "America/Los_Angeles"))
except Exception:
    # Fails open to UTC if zoneinfo's tz database isn't available in this
    # environment (rare, but possible on minimal containers) - better to
    # run with a known-approximate reset time than crash on import.
    _QUOTA_RESET_TZ = timezone.utc
    logging.getLogger(__name__).warning(
        "[STARTUP] Could not load America/Los_Angeles timezone data - "
        "falling back to UTC for daily quota reset alignment. Install "
        "tzdata if this is unexpected."
    )

logger = logging.getLogger(__name__)

# --- Batching (per-model queue now, so no cross-model filtering needed) --
BATCH_SIZE = 5  # history: 3 (original opportunistic-batching round) -> 5
# (the "quota-system audit" round, "very low impact, zero risk, zero
# latency" per that round's own testing) -> 1 (a later, since-debunked
# edit that fully disabled batching - see the header note above) -> 3
# (this file's most recent redesign only restored it partway, to 3, not
# back to the already-tested 5). This free-tier capacity audit restores
# it the rest of the way to the previously-shipped, previously-verified
# 5, since nothing about _drain_batch/_extend_batch_during_wait or the
# batch prompt/token-budget in vision.py hardcodes an assumed batch size
# - both scale off this constant and off `len(images)` dynamically, so
# this is a one-constant change, not a structural one. The real
# trade-off (worth remembering, not "zero risk" as originally framed):
# a batch's fate is shared, so a batch of 5 fails 5 jobs together
# instead of 3 on the rare dispatch that exhausts its retries - a wider
# blast radius, accepted here because it only ever matters on an
# already-rare failure path, and because higher realized batch sizes
# are the whole point when the goal is maximizing users served per day
# under real bursty load, not just steady-state throughput. At 1,
# _drain_batch()'s and _extend_batch_during_wait()'s loop bounds both
# resolve to zero, making batching fully inert (not just "rare") -
# verified this was an unintended regression, not a deliberate
# simplification. Purely opportunistic and zero-added-latency either
# way (see _drain_batch/_extend_batch_during_wait below): it only folds
# in jobs that arrive during a pacing wait the worker was already going
# to sit through, never adds a new wait of its own.

# THE REAL FIX for batching never firing in production (0 batch requests,
# 0 window joins observed over ~5 days of real traffic): the opportunistic
# reuse above only ever runs INSIDE an already-mandatory pacing wait (see
# `if wait > 0:` in _worker below, prior to this round). At light,
# realistic traffic - photos arriving minutes to hours apart, not within
# the same ~5-30s pacing window - `wait` is almost always already <= 0 by
# the time a job is picked up (the model's been idle longer than its own
# pacing interval), so there was never a mandatory wait to opportunistically
# listen during, and every job dispatched alone. This isn't a bug in
# _extend_batch_during_wait itself - it's correct for the case it handles -
# it's that the case it handles (a wait already owed) essentially never
# occurs at this traffic volume. A batch of 4 different users submitting
# within the same few seconds (the real-world case worth optimizing for)
# was structurally unable to batch even though BATCH_SIZE=5 allowed it.
#
# Fix: _worker now ALSO applies a small, unconditional, configurable
# collection window even when no mandatory pacing wait is owed - see the
# `max(wait, BATCH_WINDOW_SECONDS)` deadline in _worker below. This is a
# genuine, deliberate, bounded trade-off (unlike the zero-added-latency
# trick above): a lone user's photo can now wait up to BATCH_WINDOW_SECONDS
# longer than before. Kept intentionally short - this should be
# imperceptible next to the Telegram download + image-optimization time
# already spent per photo before a job ever reaches this queue - and fully
# configurable without a code change for operators who want to retune it.
BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "400"))
BATCH_WINDOW_SECONDS = max(BATCH_WINDOW_MS, 0) / 1000.0

# --- Adaptive pacing (AIMD) - reactive layer, combined with the
#     proactive daily-budget pacer below via max(reactive, budget_paced) --
#
# THE FIX for the real RPM bottleneck this round: BASE_INTERVAL/MIN_INTERVAL
# used to be single GLOBAL constants shared by every model in the pool,
# regardless of that model's actual RPM ceiling. That was wrong in both
# directions at once:
#   - MIN_INTERVAL=5.0s (a 12 req/min ceiling) sat ABOVE this project's own
#     documented conservative estimate of ~7-9 req/min (see README) for a
#     model like gemini-2.5-flash. The adaptive layer's clean-streak
#     "ease down" logic (SUCCESS_STREAK_FOR_DECREASE) would legitimately
#     walk the interval down to that floor after sustained health, which
#     means the pacer was *designed* to eventually pace faster than the
#     real ceiling and trigger a 429 - not bad luck, a predictable
#     consequence of the floor being wrong. This matches the reported
#     symptom exactly: RPM 429s during otherwise-healthy operation.
#   - The same shared floor also meant a materially faster model in the
#     pool (e.g. Flash-Lite, typically a higher RPM ceiling than Flash)
#     was needlessly held to Flash's conservative pace - unnecessarily
#     conservative delays, leaving real throughput on the table.
#
# Fix: derive MIN_INTERVAL/BASE_INTERVAL PER MODEL from the RPM ceiling
# configured in vision.MODEL_RPM_LIMIT (extends the existing
# GEMINI_MODEL_POOL=model:budget:rpm pattern - see vision.py). MAX_INTERVAL
# stays a single shared ceiling (a worst-case safety cap, not tied to any
# model's specific RPM number).
RPM_SAFETY_MARGIN = 0.95  # small extra buffer on top of the operator-configured
# RPM value, which is already expected to be a conservative estimate (see
# vision.MODEL_POOL comment) - this only covers rolling-window edge
# effects, not a second full safety haircut on top of that.
BASE_INTERVAL_MULTIPLIER = 1.4  # matches the original BASE/MIN ratio (7.0/5.0) -
# gives the AIMD "ease down" path room to earn its way to the safe floor
# via real clean-success streaks instead of starting there cold.
MAX_INTERVAL = 30.0
INTERVAL_INCREASE_FACTOR = 1.6
INTERVAL_DECREASE_STEP = 0.25
SUCCESS_STREAK_FOR_DECREASE = 6


def _min_interval_for_model(model: str) -> float:
    rpm = MODEL_RPM_LIMIT.get(model, DEFAULT_RPM_LIMIT)
    if rpm <= 0:
        return MAX_INTERVAL
    return 60.0 / (rpm * RPM_SAFETY_MARGIN)


def _base_interval_for_model(model: str) -> float:
    return min(_min_interval_for_model(model) * BASE_INTERVAL_MULTIPLIER, MAX_INTERVAL)

# --- Circuit breaker (safety net if the daily budget estimate is wrong) --
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0
CIRCUIT_BREAKER_MAX_COOLDOWN_SECONDS = 300.0

# --- Perceptual near-duplicate cache (modest, real, not the focus now) --
PHASH_HAMMING_THRESHOLD = 4

# --- Dynamic per-user cooldown ---------------------------------------
USER_COOLDOWN_SECONDS = 8.0
RAPID_FIRE_WINDOW_SECONDS = 300.0
RAPID_FIRE_ESCALATION_STEPS = [
    (5, 20.0),
    (10, 60.0),
    (20, 180.0),
]

# --- Other tunables --------------------------------------------------
DUPLICATE_CACHE_TTL_SECONDS = 86400  # was 3600 (1hr) - a cached analysis of a SPECIFIC photo
# never goes stale (the food in that photo doesn't change), so extending
# this to 24h only ever helps catch repeat-day patterns (e.g. the same
# breakfast photo reused across days) at zero risk and zero added latency.
DUPLICATE_CACHE_MAX_ENTRIES = 1500  # was 500 - memory cost is trivial (~KB/entry), avoids
# premature eviction now that the longer TTL means more entries accumulate.
MAX_QUEUE_SIZE_PER_MODEL = 40
STALE_USER_ENTRY_SECONDS = 3600

RATE_LIMIT_MESSAGE_KURDISH = (
    "🚦 زۆر کەس لەم کاتەدا بۆتەکە بەکاردەهێنن. "
    "تکایە چەند چرکەیەک یان خولەکێک چاوەڕێ بکە و دووبارە هەوڵبدەرەوە."
)
QUEUE_FULL_MESSAGE_KURDISH = (
    "🚦 ئێستا بۆتەکە زۆر قەرەباڵغە. تکایە دەقیقەیەک چاوەڕێ بکە و دووبارە هەوڵبدەرەوە."
)
COOLDOWN_MESSAGE_KURDISH = "⏳ تکایە چەند چرکەیەک چاوەڕێ بکە پێش ناردنی وێنەیەکی تر."
RAPID_FIRE_MESSAGE_KURDISH = (
    "⏳ لەم کاتەدا وێنەی زۆرت ناردووە. تکایە کەمێک وازبهێنە پێش ناردنی وێنەیەکی تر."
)


# --- Per-model state (module-level, single-process - fine for one Railway instance) ---

_queues: dict[str, asyncio.Queue] = {}
_worker_tasks: dict[str, asyncio.Task] = {}

_current_interval: dict[str, float] = {}
_consecutive_clean_successes: dict[str, int] = {}
_last_request_monotonic: dict[str, float] = {}

_consecutive_rate_limits: dict[str, int] = {}
_circuit_open_until: dict[str, float] = {}
_circuit_reopen_count: dict[str, int] = {}

# Daily budget tracking - the core of this round's redesign.
_daily_request_count: dict[str, int] = {}
_daily_reset_date: dict[str, str] = {}  # "YYYY-MM-DD" (UTC)


def _today_str() -> str:
    """
    THE FIX: Google resets Gemini API quotas at midnight Pacific Time,
    not UTC. Resetting our local counter at UTC midnight instead created
    a misalignment window of several hours (up to 8, depending on
    PST/PDT) where our tracker could show "new day, full budget" while
    Google's quota was still mid-cycle from the previous day (false
    abundance - could contribute to real 429s), or the reverse (false
    scarcity - needlessly self-limiting for hours after Google already
    reset). Configurable via GEMINI_QUOTA_RESET_TIMEZONE if this is ever
    wrong for a specific project/region - defaults to America/Los_Angeles
    which is the documented behavior for most Google Cloud API quotas.
    """
    return datetime.now(_QUOTA_RESET_TZ).strftime("%Y-%m-%d")


def _check_daily_reset(model: str):
    today = _today_str()
    if _daily_reset_date.get(model) != today:
        _daily_reset_date[model] = today
        _daily_request_count[model] = 0


def _remaining_daily_budget(model: str) -> int:
    _check_daily_reset(model)
    budget = MODEL_POOL.get(model, DEFAULT_DAILY_BUDGET)
    used = _daily_request_count.get(model, 0)
    return max(budget - used, 0)


def _record_daily_request(model: str, count: int = 1):
    """
    THE FIX for a real accounting bug: estimate_calories()/_batch() can
    make up to MAX_ATTEMPTS real HTTP calls internally (retries on
    transient errors), but the daily counter used to be incremented
    exactly once per JOB regardless of how many real requests that job
    actually made - undercounting true usage whenever retries happened,
    which meant the proactive budget tracker could show healthy
    headroom while the account was actually closer to its real ceiling.
    `count` should be the result's "billable_attempts" - see
    estimate_calories's docstring for why 429s are excluded from that
    count (they're typically rejected pre-processing, not quota-consuming).
    """
    _check_daily_reset(model)
    _daily_request_count[model] = _daily_request_count.get(model, 0) + max(count, 0)


def _ideal_budget_paced_interval(model: str) -> float:
    """
    THE FIX for a real production bug: this used to divide the ENTIRE
    remaining budget across the ENTIRE remaining day unconditionally -
    so even at 2/200 used, it forced a ~436s wait between every single
    request from the very first one, because 86400s / 198 remaining ≈
    436s, regardless of whether actual traffic was anywhere near that
    pace. That's backwards - "pacing should reflect actual remaining
    capacity," not a blanket rationing floor applied regardless of
    demand.

    Correct behavior: compare ACTUAL usage so far today against the
    EXPECTED usage if budget were being spent evenly across the day. If
    we're at or behind that fair-share pace (the common case for
    low-to-moderate traffic), apply NO proactive throttle at all - let
    MIN_INTERVAL and the reactive AIMD layer (which responds to real
    429s) govern pacing. Only throttle, and only proportionally, when
    we're genuinely AHEAD of pace - i.e., actually at risk of exhausting
    the day's budget early.
    """
    _check_daily_reset(model)
    budget = MODEL_POOL.get(model, DEFAULT_DAILY_BUDGET)
    if budget <= 0:
        return 0.0

    used = _daily_request_count.get(model, 0)
    remaining_budget = max(budget - used, 0)
    if remaining_budget <= 0:
        return MAX_INTERVAL  # genuinely exhausted for today

    now = datetime.now(_QUOTA_RESET_TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_elapsed_today = max((now - midnight).total_seconds(), 0.0)
    seconds_in_day = 86400.0

    expected_used_by_now = budget * (seconds_elapsed_today / seconds_in_day)

    if used <= expected_used_by_now:
        return 0.0  # at or behind fair-share pace - no proactive throttle needed

    # Ahead of pace - throttle proportionally to spread what's left of
    # the budget over what's left of the day, same idea as before but
    # only ever applied when actually warranted.
    seconds_remaining_today = max(seconds_in_day - seconds_elapsed_today, 1.0)
    return seconds_remaining_today / remaining_budget


def _interval(model: str) -> float:
    """Effective pacing interval: the SLOWER (larger) of the reactive
    AIMD interval and the proactive budget-paced interval, floored at
    this model's OWN RPM-safe floor (see _min_interval_for_model - no
    longer a single global floor shared by every model). Never faster
    than what the daily budget can sustain, never faster than the
    RPM-safe floor, but free to relax when both signals say there's
    room."""
    reactive = _current_interval.setdefault(model, _base_interval_for_model(model))
    budget_paced = _ideal_budget_paced_interval(model)
    return max(reactive, budget_paced, _min_interval_for_model(model))


def _is_circuit_open(model: str) -> bool:
    return time.monotonic() < _circuit_open_until.get(model, 0.0)


def _is_model_available(model: str) -> bool:
    return not _is_circuit_open(model) and _remaining_daily_budget(model) > 0


def select_best_model(candidate_models: list[str]) -> str:
    """
    Proactive, capacity-aware routing across the WHOLE pool: pick the
    healthy (closed circuit, budget remaining) model with the most
    PROPORTIONAL remaining daily budget - this naturally load-balances
    traffic across every model over the course of the day instead of
    draining one "preferred" model first. Falls back to whichever model
    has the soonest-reopening circuit or soonest daily reset if nothing
    is currently available - always returns something, never blocks
    submission on there being zero options.
    """
    available = [m for m in candidate_models if _is_model_available(m)]
    if available:
        def proportional_headroom(model: str) -> float:
            budget = MODEL_POOL.get(model, DEFAULT_DAILY_BUDGET)
            return _remaining_daily_budget(model) / budget if budget else 0.0
        return max(available, key=proportional_headroom)

    # Nothing available - pick whichever recovers soonest so we don't
    # get stuck permanently preferring one exhausted model.
    def recovery_eta(model: str) -> float:
        return _circuit_open_until.get(model, 0.0)
    logger.warning("[ROUTING] No model in the pool currently has budget/healthy circuit - routing to soonest-recovering")
    return min(candidate_models, key=recovery_eta) if candidate_models else list(MODEL_POOL)[0]


_last_user_request: dict[int, float] = {}
_recent_submission_times: dict[int, list[float]] = {}
_mark_call_count = 0

_result_cache: dict[str, tuple[float, dict]] = {}
_phash_index: dict[str, tuple[int, float]] = {}

_stats = {
    "total_photos_received": 0,
    "total_submitted": 0,
    "cache_hits": 0,
    "phash_cache_hits": 0,
    "batched_requests_saved": 0,
    "batch_window_joins": 0,
    "multi_photo_dispatches": 0,
    "single_photo_dispatches": 0,
    "max_batch_size": 0,
    "queue_depth_samples": 0,
    "queue_depth_sum": 0,
    "queue_depth_max": 0,
    "batching_opportunities": 0,
    "queue_full_rejections": 0,
    "cooldown_rejections": 0,
    "daily_limit_blocks": 0,
    "rapid_fire_escalations": 0,
    "successful_analyses": 0,
    "no_food_results": 0,
    "rate_limited_failures": 0,
    "other_failures": 0,
    "circuit_breaker_trips": 0,
    "circuit_breaker_short_circuits": 0,
    "started_at": time.monotonic(),
}


def _hash_image(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _get_cached(image_hash: str) -> dict | None:
    entry = _result_cache.get(image_hash)
    if entry is None:
        return None
    inserted_at, result = entry
    if time.monotonic() - inserted_at > DUPLICATE_CACHE_TTL_SECONDS:
        del _result_cache[image_hash]
        return None
    return result


def _get_cached_by_phash(dhash_value: int | None) -> dict | None:
    if dhash_value is None:
        return None
    now = time.monotonic()
    for image_hash, (stored_hash, inserted_at) in list(_phash_index.items()):
        if now - inserted_at > DUPLICATE_CACHE_TTL_SECONDS:
            del _phash_index[image_hash]
            continue
        if hamming_distance(dhash_value, stored_hash) <= PHASH_HAMMING_THRESHOLD:
            cached = _get_cached(image_hash)
            if cached is not None:
                return cached
    return None


def _store_cache(image_hash: str, result: dict, dhash_value: int | None):
    if len(_result_cache) >= DUPLICATE_CACHE_MAX_ENTRIES:
        oldest_key = min(_result_cache, key=lambda k: _result_cache[k][0])
        del _result_cache[oldest_key]
        _phash_index.pop(oldest_key, None)
    _result_cache[image_hash] = (time.monotonic(), result)
    if dhash_value is not None:
        _phash_index[image_hash] = (dhash_value, time.monotonic())


def check_user_cooldown(user_id: int) -> float:
    last = _last_user_request.get(user_id)
    base_remaining = 0.0
    if last is not None:
        base_remaining = max(USER_COOLDOWN_SECONDS - (time.monotonic() - last), 0.0)

    now = time.monotonic()
    recent = [t for t in _recent_submission_times.get(user_id, []) if now - t <= RAPID_FIRE_WINDOW_SECONDS]
    _recent_submission_times[user_id] = recent

    required_cooldown = USER_COOLDOWN_SECONDS
    for threshold_count, escalated_cooldown in RAPID_FIRE_ESCALATION_STEPS:
        if len(recent) >= threshold_count:
            required_cooldown = escalated_cooldown

    if required_cooldown > USER_COOLDOWN_SECONDS and last is not None:
        escalated_remaining = max(required_cooldown - (now - last), 0.0)
        if escalated_remaining > base_remaining:
            _stats["rapid_fire_escalations"] += 1
            logger.info(
                "[RAPID_FIRE] User has %d submissions in the last %.0fs, escalating cooldown to %.0fs",
                len(recent), RAPID_FIRE_WINDOW_SECONDS, required_cooldown,
            )
        return escalated_remaining

    return base_remaining


def mark_user_request(user_id: int):
    global _mark_call_count
    now = time.monotonic()
    _last_user_request[user_id] = now
    _recent_submission_times.setdefault(user_id, []).append(now)

    _mark_call_count += 1
    if _mark_call_count % 100 == 0:
        cutoff = now - STALE_USER_ENTRY_SECONDS
        stale = [uid for uid, t in _last_user_request.items() if t < cutoff]
        for uid in stale:
            del _last_user_request[uid]
            _recent_submission_times.pop(uid, None)


def _record_pacing_feedback(result: dict, model: str):
    """Adjusts the REACTIVE interval and circuit breaker state for THIS
    model. Combined with the proactive budget pacer in _interval(). Only
    ever called for REAL Gemini attempts - synthetic circuit-breaker
    short-circuits must never reach this function."""

    is_rate_limited = result.get("status") == "failed" and result.get("reason") == "rate_limited"

    if is_rate_limited:
        _stats["rate_limited_failures"] += 1
        _consecutive_clean_successes[model] = 0
        _consecutive_rate_limits[model] = _consecutive_rate_limits.get(model, 0) + 1
        new_interval = min(
            _current_interval.setdefault(model, _base_interval_for_model(model)) * INTERVAL_INCREASE_FACTOR,
            MAX_INTERVAL,
        )
        _current_interval[model] = new_interval
        logger.warning(
            "[ADAPTIVE_PACING] [%s] Rate limit seen, reactive interval increased to %.1fs (consecutive: %d)",
            model, new_interval, _consecutive_rate_limits[model],
        )
        if (_consecutive_rate_limits[model] >= CIRCUIT_BREAKER_THRESHOLD
                and time.monotonic() >= _circuit_open_until.get(model, 0.0)):
            reopen_count = _circuit_reopen_count.get(model, 0)
            cooldown = min(
                CIRCUIT_BREAKER_COOLDOWN_SECONDS * (2 ** reopen_count),
                CIRCUIT_BREAKER_MAX_COOLDOWN_SECONDS,
            )
            _circuit_open_until[model] = time.monotonic() + cooldown
            _circuit_reopen_count[model] = reopen_count + 1
            _stats["circuit_breaker_trips"] += 1
            logger.error(
                "[CIRCUIT_BREAKER] [%s] Opening circuit for %.0fs after %d consecutive REAL "
                "rate-limit failures (reopen #%d) - daily budget estimate may be too generous",
                model, cooldown, _consecutive_rate_limits[model], reopen_count + 1,
            )
    else:
        _consecutive_rate_limits[model] = 0
        _circuit_reopen_count[model] = 0
        if result.get("status") == "ok":
            _stats["successful_analyses"] += 1
        elif result.get("status") == "no_food":
            _stats["no_food_results"] += 1
        else:
            _stats["other_failures"] += 1

        streak = _consecutive_clean_successes.get(model, 0) + 1
        if streak >= SUCCESS_STREAK_FOR_DECREASE:
            _current_interval[model] = max(
                _current_interval.setdefault(model, _base_interval_for_model(model)) - INTERVAL_DECREASE_STEP,
                _min_interval_for_model(model),
            )
            streak = 0
            logger.info("[ADAPTIVE_PACING] [%s] Sustained clean streak, reactive interval eased to %.1fs", model, _current_interval[model])
        _consecutive_clean_successes[model] = streak


def _drain_batch(model: str) -> list[tuple]:
    """Instant, zero-cost drain of whatever's ALREADY sitting in this
    model's queue - no cross-model filtering needed now that queues are
    per-model. Called first, before any pacing wait, so a real backlog
    gets folded in immediately regardless of whether a wait follows."""
    jobs = []
    queue = _queues[model]
    while len(jobs) < BATCH_SIZE - 1:
        try:
            jobs.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return jobs


async def _extend_batch_during_wait(model: str, batch: list[tuple], wait_deadline: float) -> list[tuple]:
    """
    THE FIX for low production batching rate: the old drain only ever
    grabbed jobs that happened to already be in the queue at the exact
    instant the worker picked up job #1 - essentially never, since real
    submissions from different users/moments trickle in across a second
    or more, not within the same event-loop tick. Meanwhile the worker
    was about to sit idle for a full pacing wait (5-30s) anyway before
    the next request could legally go out.

    This keeps listening for NEW arrivals for the REMAINDER of that
    SAME mandatory pacing wait - a wait we were going to take regardless
    - instead of sleeping through it doing nothing. Zero added latency:
    job #1 (and this model's next request) were never going to leave
    before wait_deadline anyway, batched or not. Stops as soon as the
    batch is full or the deadline passes, whichever first.
    """
    queue = _queues[model]
    while len(batch) < BATCH_SIZE:
        remaining = wait_deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            job = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        batch.append(job)
    return batch


async def _worker(model: str):
    logger.info("[QUEUE] Worker started for model: %s (daily budget: %d)", model, MODEL_POOL.get(model, DEFAULT_DAILY_BUDGET))
    queue = _queues[model]
    while True:
        try:
            first_job = await queue.get()
        except Exception:
            logger.critical("[WORKER_CRASH] [%s] Failed to read from queue - retrying loop", model, exc_info=True)
            await asyncio.sleep(1)
            continue

        batch = [first_job]
        try:
            queue_depth_at_pickup = queue.qsize()
            _stats["queue_depth_samples"] += 1
            _stats["queue_depth_sum"] += queue_depth_at_pickup
            _stats["queue_depth_max"] = max(_stats["queue_depth_max"], queue_depth_at_pickup)
            if queue_depth_at_pickup > 0:
                _stats["batching_opportunities"] += 1

            more_jobs = _drain_batch(model)
            batch = [first_job] + more_jobs
            for _ in more_jobs:
                queue.task_done()

            if _is_circuit_open(model):
                logger.info("[CIRCUIT_BREAKER] [%s] Circuit open - short-circuiting %d job(s)", model, len(batch))
                _stats["circuit_breaker_short_circuits"] += len(batch)
                for image_bytes, media_type, corrections, future in batch:
                    if not future.done():
                        future.set_result({"status": "failed", "reason": "rate_limited"})
            elif _remaining_daily_budget(model) <= 0:
                logger.info("[DAILY_BUDGET] [%s] Daily budget exhausted - short-circuiting %d job(s)", model, len(batch))
                _stats["circuit_breaker_short_circuits"] += len(batch)
                for image_bytes, media_type, corrections, future in batch:
                    if not future.done():
                        future.set_result({"status": "failed", "reason": "rate_limited"})
            else:
                elapsed = time.monotonic() - _last_request_monotonic.get(model, 0.0)
                wait = _interval(model) - elapsed

                # Collection window: run this whenever the batch still has
                # room, NOT only when a mandatory pacing wait happens to
                # already be owed (wait > 0) - see the comment on
                # BATCH_WINDOW_MS above for why that old gate is exactly
                # why batching never fired in production. When wait > 0,
                # this costs nothing extra (same as before: we piggyback on
                # a wait we owed anyway). When wait <= 0, this is a real,
                # deliberate, bounded wait of up to BATCH_WINDOW_SECONDS -
                # the one genuinely new trade-off in this fix.
                size_before_window = len(batch)
                if size_before_window < BATCH_SIZE:
                    collection_deadline = time.monotonic() + max(wait, BATCH_WINDOW_SECONDS)
                    logger.info(
                        "[QUEUE] [%s] Collecting batch-mates for up to %.2fs "
                        "(mandatory pacing wait: %.1fs)", model, max(wait, BATCH_WINDOW_SECONDS), wait,
                    )
                    batch = await _extend_batch_during_wait(model, batch, collection_deadline)
                    joined = len(batch) - size_before_window
                    if joined > 0:
                        _stats["batch_window_joins"] += joined
                        for _ in range(joined):
                            queue.task_done()
                # Whatever's left of the mandatory pacing wait after the
                # collection window above (should be ~0 in practice - the
                # loop only stops early if the batch filled up before the
                # deadline, in which case we still owe the rest of the
                # interval before the request can legally go out). When
                # wait was already <= 0 going in, this is also <= 0 and a
                # no-op - the collection window above was the only wait.
                remaining = _interval(model) - (time.monotonic() - _last_request_monotonic.get(model, 0.0))
                if remaining > 0:
                    await asyncio.sleep(remaining)

                if len(batch) == 1:
                    image_bytes, media_type, corrections, future = batch[0]
                    try:
                        result = await asyncio.to_thread(estimate_calories, image_bytes, media_type, corrections, model=model)
                    except Exception:
                        logger.exception("[QUEUE] [%s] Unhandled exception from estimate_calories", model)
                        result = {"status": "failed", "reason": "other"}

                    _last_request_monotonic[model] = time.monotonic()
                    _record_daily_request(model, result.get("billable_attempts", 1))
                    _record_pacing_feedback(result, model)
                    _stats["single_photo_dispatches"] += 1
                    _stats["max_batch_size"] = max(_stats["max_batch_size"], 1)
                    if not future.done():
                        future.set_result(result)
                else:
                    logger.info("[BATCH] [%s] Processing %d photos in ONE request", model, len(batch))
                    _stats["batched_requests_saved"] += len(batch) - 1
                    _stats["multi_photo_dispatches"] += 1
                    _stats["max_batch_size"] = max(_stats["max_batch_size"], len(batch))
                    images = [(img, mt) for img, mt, _, _ in batch]
                    corrections = batch[0][2]
                    try:
                        batch_result = await asyncio.to_thread(estimate_calories_batch, images, corrections, model=model)
                    except Exception:
                        logger.exception("[QUEUE] [%s] Unhandled exception from estimate_calories_batch", model)
                        batch_result = {"status": "failed", "reason": "other"}

                    _last_request_monotonic[model] = time.monotonic()
                    _record_daily_request(model, batch_result.get("billable_attempts", 1))  # real attempts, not job count
                    _record_pacing_feedback(batch_result, model)

                    if batch_result["status"] == "ok":
                        for (_, _, _, future), result in zip(batch, batch_result["results"]):
                            if not future.done():
                                future.set_result(result)
                    else:
                        for _, _, _, future in batch:
                            if not future.done():
                                future.set_result(batch_result)

        except Exception:
            logger.critical("[WORKER_CRASH] [%s] Unexpected error processing batch", model, exc_info=True)
            try:
                for image_bytes, media_type, corrections, future in batch:
                    if not future.done():
                        future.set_result({"status": "failed", "reason": "other"})
            except Exception:
                pass
        finally:
            queue.task_done()


def start_workers():
    """Call once, from inside a running event loop (e.g. PTB's post_init).
    Starts one dedicated queue + worker PER MODEL in the pool - true
    parallelism between models, not one shared sequential worker."""
    for model in MODEL_POOL:
        if model in _worker_tasks:
            continue
        _queues[model] = asyncio.Queue()
        _worker_tasks[model] = asyncio.create_task(_worker(model))
    logger.info("[STARTUP] Started %d worker(s) for pool: %s", len(_worker_tasks), list(MODEL_POOL.keys()))


async def submit_photo_job(
    image_bytes: bytes, media_type: str, corrections: list[dict],
    candidate_models: list[str] | None = None,
) -> tuple[dict, int, int | None]:
    """
    Enqueues an ALREADY-OPTIMIZED, ALREADY-LOCALLY-VALIDATED photo.
    candidate_models defaults to the full configured pool - the router
    picks whichever healthy model currently has the most proportional
    daily headroom (see select_best_model). Pass a narrower list only if
    you have a specific reason to restrict routing for this request.
    Returns (result_dict, queue_position, dhash_value).
    """
    _stats["total_submitted"] += 1
    candidate_models = candidate_models or list(MODEL_POOL.keys())

    image_hash = _hash_image(image_bytes)
    cached = _get_cached(image_hash)
    if cached is not None:
        logger.info("[CACHE] Exact duplicate image detected, reusing cached result")
        _stats["cache_hits"] += 1
        return cached, 0, None

    dhash_value = compute_dhash(image_bytes)
    near_cached = _get_cached_by_phash(dhash_value)
    if near_cached is not None:
        logger.info("[PHASH] Near-duplicate image detected, reusing cached result")
        _stats["phash_cache_hits"] += 1
        _store_cache(image_hash, near_cached, dhash_value)
        return near_cached, 0, dhash_value

    if not _worker_tasks:
        start_workers()

    model = select_best_model(candidate_models)
    queue = _queues[model]

    if queue.qsize() >= MAX_QUEUE_SIZE_PER_MODEL:
        logger.warning("[QUEUE] [%s] Rejecting submission - queue full (%d items)", model, queue.qsize())
        _stats["queue_full_rejections"] += 1
        return {"status": "failed", "reason": "queue_full"}, queue.qsize(), dhash_value

    queue_position = queue.qsize()

    future = asyncio.get_running_loop().create_future()
    await queue.put((image_bytes, media_type, corrections, future))
    result = await future

    if result.get("status") == "ok":
        _store_cache(image_hash, result, dhash_value)

    return result, queue_position, dhash_value


def record_photo_received():
    _stats["total_photos_received"] += 1


def record_cooldown_block():
    _stats["cooldown_rejections"] += 1


def record_daily_limit_block():
    _stats["daily_limit_blocks"] += 1


def get_stats_summary() -> dict:
    uptime_seconds = time.monotonic() - _stats["started_at"]
    total = _stats["total_submitted"]
    cache_hit_rate = ((_stats["cache_hits"] + _stats["phash_cache_hits"]) / total * 100) if total else 0.0
    avg_queue_depth = (
        _stats["queue_depth_sum"] / _stats["queue_depth_samples"]
        if _stats["queue_depth_samples"] else 0.0
    )
    per_model_health = {
        model: {
            "pacing_interval": round(_interval(model), 1),
            "rpm_limit": MODEL_RPM_LIMIT.get(model, DEFAULT_RPM_LIMIT),
            "circuit_open": _is_circuit_open(model),
            "daily_budget": MODEL_POOL.get(model, DEFAULT_DAILY_BUDGET),
            "daily_used": _daily_request_count.get(model, 0) if _daily_reset_date.get(model) == _today_str() else 0,
            "daily_remaining": _remaining_daily_budget(model),
        }
        for model in MODEL_POOL
    }
    total_queue_depth = sum(q.qsize() for q in _queues.values())

    # Batching telemetry: the key capacity metric is photos-per-real-
    # Gemini-request, since that (not raw job count) is what the daily
    # dispatch budget actually limits. All derived from existing counters
    # except multi_photo_dispatches/single_photo_dispatches/max_batch_size,
    # which are tracked directly at the two dispatch sites in _worker().
    real_gemini_requests = (
        _stats["successful_analyses"] + _stats["no_food_results"]
        + _stats["rate_limited_failures"] + _stats["other_failures"]
    )
    total_batches = _stats["multi_photo_dispatches"]
    photos_in_batches = _stats["batched_requests_saved"] + total_batches
    total_photos_dispatched = real_gemini_requests + _stats["batched_requests_saved"]
    avg_batch_size = round(total_photos_dispatched / real_gemini_requests, 2) if real_gemini_requests else 0.0
    batch_utilization_pct = round(avg_batch_size / BATCH_SIZE * 100, 1) if BATCH_SIZE else 0.0

    return {
        **_stats,
        **get_http_stats(),  # additive only - raw_http_attempts/429_responses/
        # successes/other_failures, distinct from and never overwriting any of the
        # dispatch-level counters already in _stats above (see vision.py's
        # _gemini_rest_request for why these are tracked separately).
        "uptime_minutes": round(uptime_seconds / 60, 1),
        "cache_hit_rate_pct": round(cache_hit_rate, 1),
        "queue_depth": total_queue_depth,
        "avg_queue_depth_at_pickup": round(avg_queue_depth, 2),
        "per_model_health": per_model_health,
        "cached_entries": len(_result_cache),
        "tracked_users": len(_last_user_request),
        "real_gemini_requests": real_gemini_requests,
        "total_batches": total_batches,
        "photos_in_batches": photos_in_batches,
        "total_photos_dispatched": total_photos_dispatched,
        "avg_photos_per_request": avg_batch_size,
        "batch_utilization_pct": batch_utilization_pct,
        "batch_window_ms": BATCH_WINDOW_MS,
    }
