# storage.py
#
# Minimal SQLite storage - one file, no server needed, no paid services.
# Good enough for an MVP up to several thousand users.
#
# Tables:
#   meals       - one row per photo analyzed (unchanged from before)
#   users       - profile, targets, and onboarding/correction conversation
#                 state (kept in the DB, not memory, so it survives restarts)
#   corrections - user-submitted fixes, reused as extra glossary context
#                 for every future photo (this is the "free learning" loop)
#
# All ALTER TABLE migrations are additive and safe to run on top of an
# already-deployed database - existing data is never touched or dropped.

import json
import sqlite3
from datetime import datetime, timedelta

DB_PATH = "calories.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    # WAL mode lets reads and writes happen concurrently instead of
    # blocking each other - free improvement for multiple simultaneous
    # users, no behavior change otherwise.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _add_missing_columns(conn, table: str, columns: dict):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col, col_type in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_db():
    conn = _connect()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            food_name_kurdish TEXT,
            food_name_english TEXT,
            kcal INTEGER,
            protein_g INTEGER,
            carbs_g INTEGER,
            fat_g INTEGER,
            confidence TEXT,
            matched_glossary INTEGER,
            feedback TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _add_missing_columns(
        conn, "meals",
        {"foods_json": "TEXT", "note_kurdish": "TEXT", "insight_kurdish": "TEXT",
         "dhash": "INTEGER", "model_used": "TEXT"},
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            height_cm REAL,
            weight_kg REAL,
            goal TEXT,
            activity_level TEXT,
            bmr INTEGER,
            tdee INTEGER,
            target_kcal INTEGER,
            target_protein_g INTEGER,
            target_carbs_g INTEGER,
            target_fat_g INTEGER,
            onboarding_step TEXT,
            pending_correction_meal_id INTEGER,
            created_at TEXT
        )
        """
    )
    # 'tier' controls the daily analysis limit (see DAILY_LIMITS below).
    # Default 'free' for everyone existing and new - premium is added by
    # just changing this value for a user, no schema change needed later.
    _add_missing_columns(conn, "users", {"tier": "TEXT DEFAULT 'free'"})

    # Item-by-item correction flow state (Stage 2). pending_correction_meal_id
    # (already existed) now means "this meal is under item-by-item review";
    # these two new columns track exactly where in that review the user is.
    # Kept in the DB, not memory, same reasoning as onboarding state above -
    # survives a restart mid-correction instead of leaving a user stuck.
    _add_missing_columns(
        conn, "users",
        {
            "pending_correction_food_index": "INTEGER",
            "pending_correction_awaiting_text": "INTEGER DEFAULT 0",
        },
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            wrong_name TEXT,
            correct_name_kurdish TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    # Links a correction back to the photo's fingerprint (see
    # analysis_fingerprints below) so future training can distinguish
    # "Gemini got this right" from "Gemini got this wrong, corrected to X"
    # - a much stronger training signal than uncorrected examples alone.
    _add_missing_columns(conn, "corrections", {"dhash": "INTEGER"})

    # --- Kurdish correction memory (Stage 2) -----------------------------
    #
    # THE PROBLEM THIS FIXES: the old `corrections` table above fed EVERY
    # submitted correction into EVERY future Gemini prompt via
    # get_all_corrections(), with no moderation at all - one user typing
    # garbage or a joke would immediately become "reliable signal" shown
    # to Gemini for every other user's photos too. `corrections` is kept
    # for backward compatibility (existing rows, existing call sites) but
    # is no longer what feeds the prompt - see get_trusted_aliases_for_prompt.
    #
    # food_aliases: one row per DISTINCT (observed_name, canonical_name)
    # pair ever proposed. status starts at 'candidate' and is promoted
    # based on how many DIFFERENT users independently proposed the SAME
    # pair - never based on one user repeating themselves (see
    # food_alias_votes below, which is what makes distinct-user counting
    # possible and prevents exactly that gaming vector).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS food_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_name TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            food_id TEXT,
            confirmation_count INTEGER NOT NULL DEFAULT 1,
            distinct_user_count INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'candidate',
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            UNIQUE(observed_name, canonical_name)
        )
        """
    )
    # Junction table: which users have voted for which alias. The PRIMARY
    # KEY (not just an index) is what makes a second vote from the SAME
    # user a true no-op (INSERT OR IGNORE) rather than something that
    # needs application-code deduplication - a single user spamming the
    # same correction 100 times still only ever counts as ONE distinct
    # voter, straight from the schema, not from careful call-site logic
    # that could be gotten wrong later.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS food_alias_votes (
            alias_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (alias_id, user_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_food_aliases_status ON food_aliases(status)"
    )

    # Foundation for a POSSIBLE future local-recognition layer (see
    # README) - logs every successful analysis's dHash fingerprint
    # alongside what Gemini identified. Purely additive, write-only from
    # the app's perspective right now - nothing reads this table to skip
    # a Gemini call yet. Deliberately deferred: with zero accumulated
    # data today, any bypass logic built on this would either rarely
    # trigger (no real savings) or be forced to use a loose-enough
    # threshold to get coverage, which risks silently serving wrong
    # nutrition data. Revisit once this table has real volume.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dhash INTEGER,
            food_summary TEXT,
            total_kcal INTEGER,
            confidence TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _add_missing_columns(
        conn, "analysis_fingerprints",
        {
            # Full structured per-food data (name, portion, macros,
            # matched_glossary), not just a summary string - this is what
            # actually makes the table useful as future training labels
            # rather than a rough log.
            "foods_json": "TEXT",
            # How many of the detected foods matched the static glossary.
            # Rows with 0 are the most valuable to review later - they
            # represent foods Gemini identified that aren't in our
            # glossary yet, i.e. real gaps in local coverage.
            "matched_glossary_count": "INTEGER DEFAULT 0",
            # Flipped to 1 by mark_fingerprint_corrected() when a user
            # later corrects this exact photo - a stronger training
            # signal than an uncorrected example.
            "was_corrected": "INTEGER DEFAULT 0",
        },
    )

    # Every /today, /week, /month, /history query filters on exactly this
    # combination. Free at this row count, essential once it grows -
    # without it, these become full table scans.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_meals_user_created ON meals(user_id, created_at)"
    )
    # Partial index - only rows with feedback set are ever queried this way
    # (review_feedback.py), so indexing only those keeps it small and cheap.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_meals_feedback ON meals(feedback) WHERE feedback IS NOT NULL"
    )

    conn.commit()
    conn.close()


def log_analysis_fingerprint(dhash_value: int | None, result: dict):
    """
    Write-only logging for the future local-recognition foundation
    described above. Never raises - this is pure observability, a
    failure here must never affect the actual user-facing flow.
    """
    if dhash_value is None:
        return
    try:
        foods = result.get("foods", [])
        food_summary = "، ".join(f["name_kurdish"] for f in foods)
        matched_count = sum(1 for f in foods if f.get("matched_glossary"))
        conn = _connect()
        conn.execute(
            """
            INSERT INTO analysis_fingerprints
                (dhash, food_summary, total_kcal, confidence, created_at,
                 foods_json, matched_glossary_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (dhash_value, food_summary, result.get("total_kcal", 0),
             result.get("confidence", ""), datetime.utcnow().isoformat(),
             json.dumps(foods, ensure_ascii=False), matched_count),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # never let logging break the real flow


def mark_fingerprint_corrected(dhash_value: int | None):
    """Called when a user corrects a meal - flags the matching
    fingerprint(s) as corrected, a stronger training signal than an
    uncorrected example. Never raises, same reasoning as above."""
    if dhash_value is None:
        return
    try:
        conn = _connect()
        conn.execute(
            "UPDATE analysis_fingerprints SET was_corrected = 1 WHERE dhash = ?",
            (dhash_value,),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# --- Meals ---------------------------------------------------------------

def log_meal(user_id: int, result: dict, dhash_value: int | None = None) -> int:
    """Inserts a meal (one row per photo) and returns its row id."""

    foods = result.get("foods", [])
    food_summary = "، ".join(f["name_kurdish"] for f in foods) or "نەناسراو"
    any_matched = any(f.get("matched_glossary") for f in foods)

    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO meals (user_id, food_name_kurdish, food_name_english,
                            kcal, protein_g, carbs_g, fat_g, confidence,
                            matched_glossary, feedback, created_at,
                            foods_json, note_kurdish, insight_kurdish, dhash, model_used)
        VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, food_summary,
            result.get("total_kcal", 0), result.get("total_protein_g", 0),
            result.get("total_carbs_g", 0), result.get("total_fat_g", 0),
            result.get("confidence", "نزم"), int(bool(any_matched)),
            datetime.utcnow().isoformat(),
            json.dumps(foods, ensure_ascii=False),
            result.get("note_kurdish", ""), result.get("insight_kurdish", ""),
            dhash_value, result.get("model_used", ""),
        ),
    )
    conn.commit()
    meal_id = cur.lastrowid
    conn.close()
    return meal_id


def get_model_ab_comparison() -> list[dict]:
    """Per-model breakdown for A/B testing: count, avg confidence-is-high
    rate, and correction rate. This is the real comparative data - can't
    be gathered any other way without calling the live API myself."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT
            model_used,
            COUNT(*) AS total_meals,
            SUM(CASE WHEN confidence = 'بەرز' THEN 1 ELSE 0 END) AS high_confidence_count,
            SUM(CASE WHEN feedback = 'wrong' THEN 1 ELSE 0 END) AS wrong_feedback_count
        FROM meals
        WHERE model_used IS NOT NULL AND model_used != ''
        GROUP BY model_used
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def set_feedback(meal_id: int, feedback: str):
    conn = _connect()
    conn.execute("UPDATE meals SET feedback = ? WHERE id = ?", (feedback, meal_id))
    conn.commit()
    conn.close()


def get_meal(meal_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM meals WHERE id = ?", (meal_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_wrong_feedback_log(limit: int = 100) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM meals WHERE feedback = 'wrong' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _totals_between(user_id: int, start: datetime, end: datetime) -> dict:
    conn = _connect()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(kcal), 0), COALESCE(SUM(protein_g), 0),
               COALESCE(SUM(carbs_g), 0), COALESCE(SUM(fat_g), 0), COUNT(*)
        FROM meals WHERE user_id = ? AND created_at >= ? AND created_at < ?
        """,
        (user_id, start.isoformat(), end.isoformat()),
    ).fetchone()
    conn.close()
    kcal, protein, carbs, fat, count = row
    return {"kcal": kcal, "protein_g": protein, "carbs_g": carbs, "fat_g": fat, "meal_count": count}


def get_today_total(user_id: int) -> dict:
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return _totals_between(user_id, start, start + timedelta(days=1))


def get_week_total(user_id: int) -> dict:
    start = datetime.utcnow() - timedelta(days=7)
    return _totals_between(user_id, start, datetime.utcnow() + timedelta(minutes=1))


def get_month_total(user_id: int) -> dict:
    start = datetime.utcnow() - timedelta(days=30)
    return _totals_between(user_id, start, datetime.utcnow() + timedelta(minutes=1))


def get_meal_count_today(user_id: int) -> int:
    return get_today_total(user_id)["meal_count"]


# --- Daily usage limits --------------------------------------------------
#
# Counts SUCCESSFUL analyses only - meals only ever gets a row when
# status == "ok" (verified: failed/rate-limited/no-food results are never
# logged, see bot.py), so a user never loses quota to a bad photo, a busy
# queue, or a cache hit not counting against them unfairly... actually a
# cache hit DOES still log a meal (it's a real successful result, just
# free to serve) - that's correct and intentional, it's still "their"
# meal being tracked, the cache only saves the Gemini call, not the quota
# slot. This is a deliberate design choice: the limit protects Gemini
# quota from HEAVY USERS, not from the bot's own cache efficiency.
#
# Designed for trivial premium extension: tiers are a dict lookup and a
# user's tier is a single column - upgrading someone is one UPDATE
# statement, no schema change, no migration needed later.
DAILY_LIMITS = {
    "free": 4,
    "premium": 50,  # generous, not truly unlimited - still protects shared quota
}

LIMIT_REACHED_MESSAGE_KURDISH = (
    "📊 ئەمڕۆ گەیشتیتە سنووری {used}/{limit} شیکردنەوەی خۆراک بۆ به‌کارهێنه‌ری "
    "بێبەرامبەر.\n\nسبەی دووبارە دەتوانیت بەکاریبهێنیت، یان چاوەڕێی وردەکاری "
    "premium بکە بۆ سنووری زیاتر."
)


def get_daily_limit(tier: str) -> int:
    return DAILY_LIMITS.get(tier, DAILY_LIMITS["free"])


# --- Atomic reservation (fixes a real race condition) ---------------------
#
# BUG THIS FIXES: check_daily_limit() alone reads the count of meals
# already LOGGED in the DB - but a meal is only logged after the full
# Gemini round-trip completes (several seconds). With concurrent_updates=
# True (needed for other good reasons - see bot.py), multiple photos from
# the SAME user sent rapidly run as concurrent handlers. All of them could
# read the same "4 used" count and all pass the check before any of their
# meals are actually written back - letting more requests through than
# the limit allows and consuming real Gemini quota for it.
#
# Fix: reserve_daily_slot() checks AND increments in one synchronous call
# with no `await` inside it. Asyncio can only switch between coroutines at
# an `await` point, so this function body can never be interleaved by
# another concurrent handler - the check-then-increment is now atomic.
# release_daily_slot() must be called once the real outcome is known
# (success or failure) so a failed/no-food analysis doesn't unfairly cost
# the user a slot, and so the reservation doesn't double-count once the
# real meal row exists in the DB.
_daily_reservations: dict[int, tuple[str, int]] = {}


def _today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


_reservation_op_count = 0


def _cleanup_stale_reservations():
    """Same opportunistic-cleanup pattern as gemini_queue's cooldown
    tracking - removes entries from previous days with nothing in-flight,
    so this dict doesn't grow forever across the app's lifetime."""
    today = _today_str()
    stale = [
        uid for uid, (date_str, count) in _daily_reservations.items()
        if date_str != today and count == 0
    ]
    for uid in stale:
        del _daily_reservations[uid]


def reserve_daily_slot(user_id: int) -> tuple[bool, int, int]:
    """Returns (reserved, used_after_this_reservation, limit)."""
    global _reservation_op_count
    _reservation_op_count += 1
    if _reservation_op_count % 200 == 0:
        _cleanup_stale_reservations()

    today = _today_str()
    date_str, reserved_count = _daily_reservations.get(user_id, (today, 0))
    if date_str != today:
        reserved_count = 0  # new day

    committed = get_meal_count_today(user_id)  # already-logged meals
    total_used = committed + reserved_count     # + in-flight, not yet logged

    user = get_user(user_id)
    tier = (user or {}).get("tier") or "free"
    limit = get_daily_limit(tier)

    if total_used >= limit:
        return False, total_used, limit

    _daily_reservations[user_id] = (today, reserved_count + 1)
    return True, total_used + 1, limit


def release_daily_slot(user_id: int):
    """Call after the outcome is known, success or failure - see the
    docstring above for why this must always be called exactly once per
    successful reserve_daily_slot() call."""
    today = _today_str()
    date_str, reserved_count = _daily_reservations.get(user_id, (today, 0))
    if date_str == today and reserved_count > 0:
        _daily_reservations[user_id] = (today, reserved_count - 1)


def check_daily_limit(user_id: int) -> tuple[bool, int, int]:
    """Read-only status check (for /today display) - does NOT reserve a
    slot. Use reserve_daily_slot() before actually submitting a photo."""
    committed = get_meal_count_today(user_id)
    _, reserved_count = _daily_reservations.get(user_id, (_today_str(), 0))
    used = committed + reserved_count
    user = get_user(user_id)
    tier = (user or {}).get("tier") or "free"
    limit = get_daily_limit(tier)
    return used < limit, used, limit


def set_user_tier(user_id: int, tier: str):
    """The entire 'upgrade to premium' operation, today or in the future."""
    ensure_user(user_id)
    update_user_fields(user_id, tier=tier)


def get_recent_meals(user_id: int, limit: int = 10) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM meals WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def reset_user_history(user_id: int):
    conn = _connect()
    conn.execute("DELETE FROM meals WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# --- User profile & onboarding state -------------------------------------

def get_user(user_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def ensure_user(user_id: int):
    """Creates a blank user row if one doesn't exist yet. Safe to call anytime."""
    conn = _connect()
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
        (user_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def set_onboarding_step(user_id: int, step: str | None):
    ensure_user(user_id)
    conn = _connect()
    conn.execute(
        "UPDATE users SET onboarding_step = ? WHERE user_id = ?", (step, user_id)
    )
    conn.commit()
    conn.close()


def update_user_fields(user_id: int, **fields):
    ensure_user(user_id)
    conn = _connect()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE users SET {set_clause} WHERE user_id = ?",
        (*fields.values(), user_id),
    )
    conn.commit()
    conn.close()


def has_complete_profile(user_id: int) -> bool:
    user = get_user(user_id)
    return bool(user and user.get("target_kcal"))


def set_pending_correction(user_id: int, meal_id: int | None):
    ensure_user(user_id)
    conn = _connect()
    conn.execute(
        "UPDATE users SET pending_correction_meal_id = ? WHERE user_id = ?",
        (meal_id, user_id),
    )
    conn.commit()
    conn.close()


# --- Item-by-item correction flow state (Stage 2) -------------------------
#
# Three columns together describe where a user is in the flow:
#   pending_correction_meal_id    - which meal (or NULL if not correcting)
#   pending_correction_food_index - which item within that meal's foods
#                                    list is currently being reviewed
#   pending_correction_awaiting_text - 0 while showing yes/no buttons for
#                                    the current item, 1 after the user
#                                    tapped "no" and we're waiting for
#                                    them to type the correct name
# handle_text (bot.py) checks awaiting_text specifically so a stray text
# message sent WHILE the yes/no buttons are showing (not expected) gets a
# gentle "use the buttons" nudge instead of being silently misinterpreted.

def start_item_correction(user_id: int, meal_id: int):
    ensure_user(user_id)
    conn = _connect()
    conn.execute(
        "UPDATE users SET pending_correction_meal_id = ?, "
        "pending_correction_food_index = 0, pending_correction_awaiting_text = 0 "
        "WHERE user_id = ?",
        (meal_id, user_id),
    )
    conn.commit()
    conn.close()


def advance_correction_index(user_id: int, next_index: int):
    conn = _connect()
    conn.execute(
        "UPDATE users SET pending_correction_food_index = ?, "
        "pending_correction_awaiting_text = 0 WHERE user_id = ?",
        (next_index, user_id),
    )
    conn.commit()
    conn.close()


def set_awaiting_text(user_id: int, awaiting: bool):
    conn = _connect()
    conn.execute(
        "UPDATE users SET pending_correction_awaiting_text = ? WHERE user_id = ?",
        (int(awaiting), user_id),
    )
    conn.commit()
    conn.close()


def clear_item_correction(user_id: int):
    conn = _connect()
    conn.execute(
        "UPDATE users SET pending_correction_meal_id = NULL, "
        "pending_correction_food_index = NULL, pending_correction_awaiting_text = 0 "
        "WHERE user_id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


# --- Meal foods access (needed to correct ONE item, not re-analyze) -------

def get_meal_foods(meal_id: int) -> list[dict]:
    """Parses the meal's stored foods_json - this is the per-item detail
    the item-by-item correction flow reads and writes, distinct from the
    summary columns (food_name_kurdish, kcal, ...) which stay in sync via
    update_meal_foods below."""
    meal = get_meal(meal_id)
    if not meal or not meal.get("foods_json"):
        return []
    try:
        return json.loads(meal["foods_json"])
    except (json.JSONDecodeError, TypeError):
        return []


def update_meal_foods(meal_id: int, foods: list[dict]):
    """Rewrites a meal's per-item foods_json AND recomputes the summary
    columns (kcal/protein_g/carbs_g/fat_g/food_name_kurdish) that /today,
    /week, /month, and /history all read directly - a correction that
    only updated foods_json without this would silently desync a user's
    stats from what they see on the corrected result message."""
    food_summary = "، ".join(f.get("name_kurdish", "؟") for f in foods) or "نەناسراو"
    any_matched = any(f.get("matched_glossary") for f in foods)
    conn = _connect()
    conn.execute(
        """
        UPDATE meals SET
            foods_json = ?, food_name_kurdish = ?, matched_glossary = ?,
            kcal = ?, protein_g = ?, carbs_g = ?, fat_g = ?
        WHERE id = ?
        """,
        (
            json.dumps(foods, ensure_ascii=False), food_summary, int(bool(any_matched)),
            sum(f.get("kcal", 0) for f in foods), sum(f.get("protein_g", 0) for f in foods),
            sum(f.get("carbs_g", 0) for f in foods), sum(f.get("fat_g", 0) for f in foods),
            meal_id,
        ),
    )
    conn.commit()
    conn.close()


# --- Corrections (the free "learning" loop) -------------------------------

def save_correction(user_id: int, wrong_name: str, correct_name_kurdish: str, dhash_value: int | None = None):
    conn = _connect()
    conn.execute(
        """
        INSERT INTO corrections (user_id, wrong_name, correct_name_kurdish, created_at, dhash)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, wrong_name, correct_name_kurdish, datetime.utcnow().isoformat(), dhash_value),
    )
    conn.commit()
    conn.close()
    mark_fingerprint_corrected(dhash_value)


def get_all_corrections(limit: int = 200) -> list[dict]:
    """
    Returns recent corrections across all users, UNFILTERED by
    confirmation - kept only for backward compatibility (existing rows,
    e.g. review_feedback.py or a future audit). NOT used to build the
    Gemini prompt anymore - see get_trusted_aliases_for_prompt below for
    why an unmoderated feed was a real problem.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT DISTINCT wrong_name, correct_name_kurdish FROM corrections "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Kurdish correction memory: candidate -> supported -> trusted --------
#
# THE PROBLEM: get_all_corrections() above feeds EVERY correction ever
# submitted into EVERY future Gemini prompt (see vision.py's
# _build_corrections_prompt), with no check at all on whether it's a
# genuine fix or one user's joke/garbage input. A single bad-faith
# correction would immediately become "reliable signal" shown to Gemini
# for every other user's photos.
#
# THE FIX: every correction is recorded here first, and only promoted to
# 'trusted' (the status that actually reaches the prompt - see
# get_trusted_aliases_for_prompt) once multiple DIFFERENT users have
# independently proposed the exact same fix. A single user, however many
# times they resubmit the same text, can only ever supply ONE vote -
# distinct_user_count is backed by food_alias_votes' PRIMARY KEY
# (alias_id, user_id), not application logic that could be gotten wrong.
#
# Matching is exact-string (after light normalization), not fuzzy - two
# users spelling the same correction slightly differently are tracked as
# separate candidates rather than reinforcing each other. A deliberate,
# documented trade-off for reliability/predictability over coverage; a
# future fuzzy-merge pass is a reasonable enhancement, not implemented
# here (see kurdcalobot-128mb's "smallest safe changes" instruction).
SUPPORTED_THRESHOLD = 2  # distinct users
TRUSTED_THRESHOLD = 3    # distinct users
CANDIDATE_EXPIRY_DAYS = 30  # a 'candidate' with only 1 voter this old is cleaned up


def _normalize_alias_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def record_alias_correction(
    user_id: int, observed_name: str, canonical_name: str, food_id: str | None = None
) -> dict:
    """
    Records one user's correction as a vote toward promoting (or creating)
    an alias candidate. Returns {"status": ..., "distinct_user_count": ...}
    so the caller can tell the user something honest ("noted" vs. "this is
    now a trusted local name") instead of implying every correction is
    immediately authoritative.

    Never raises - a failure here must never break the user-visible
    correction flow, which has already applied their correction to
    THEIR OWN meal by the time this is called (see bot.py) regardless of
    whether it earns global trust.
    """
    global _alias_cleanup_op_count
    _alias_cleanup_op_count += 1
    if _alias_cleanup_op_count % 20 == 0:
        # Same opportunistic-cleanup pattern as reserve_daily_slot's
        # _cleanup_stale_reservations - runs every 20th correction rather
        # than on every write, cheap and self-contained (no separate
        # scheduler needed on a 128 MB box).
        cleanup_stale_alias_candidates()

    observed = _normalize_alias_text(observed_name)
    canonical = _normalize_alias_text(canonical_name)
    if not observed or not canonical:
        return {"status": "candidate", "distinct_user_count": 0}

    try:
        now = datetime.utcnow().isoformat()
        conn = _connect()
        row = conn.execute(
            "SELECT id FROM food_aliases WHERE observed_name = ? AND canonical_name = ?",
            (observed, canonical),
        ).fetchone()

        if row is None:
            cur = conn.execute(
                """
                INSERT INTO food_aliases
                    (observed_name, canonical_name, food_id, confirmation_count,
                     distinct_user_count, status, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, 1, 1, 'candidate', ?, ?)
                """,
                (observed, canonical, food_id, now, now),
            )
            alias_id = cur.lastrowid
        else:
            alias_id = row["id"]
            conn.execute(
                "UPDATE food_aliases SET confirmation_count = confirmation_count + 1, "
                "last_seen_at = ?, food_id = COALESCE(food_id, ?) WHERE id = ?",
                (now, food_id, alias_id),
            )

        conn.execute(
            "INSERT OR IGNORE INTO food_alias_votes (alias_id, user_id, created_at) VALUES (?, ?, ?)",
            (alias_id, user_id, now),
        )
        distinct_count = conn.execute(
            "SELECT COUNT(*) FROM food_alias_votes WHERE alias_id = ?", (alias_id,)
        ).fetchone()[0]

        if distinct_count >= TRUSTED_THRESHOLD:
            new_status = "trusted"
        elif distinct_count >= SUPPORTED_THRESHOLD:
            new_status = "supported"
        else:
            new_status = "candidate"
        conn.execute(
            "UPDATE food_aliases SET distinct_user_count = ?, status = ? WHERE id = ?",
            (distinct_count, new_status, alias_id),
        )

        conn.commit()
        conn.close()
        return {"status": new_status, "distinct_user_count": distinct_count}
    except Exception:
        return {"status": "candidate", "distinct_user_count": 0}


def get_trusted_aliases_for_prompt(limit: int = 60) -> list[dict]:
    """
    Replaces get_all_corrections() as the Gemini-prompt feed - returns
    ONLY 'trusted' aliases (multiple independent users confirmed the same
    fix), shaped with the SAME keys vision.py's _build_corrections_prompt
    already reads ("wrong_name", "correct_name_kurdish") so no change was
    needed there - just which storage function bot.py calls.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT observed_name AS wrong_name, canonical_name AS correct_name_kurdish "
        "FROM food_aliases WHERE status = 'trusted' ORDER BY last_seen_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_resolved_alias_food_id(canonical_name: str) -> str | None:
    """
    Second-tier LOCAL resolution step for a typed correction, checked
    AFTER the static glossary (knowledge.find_glossary_match) comes up
    empty - see bot.py's _build_corrected_food. Returns a food_id only
    for a 'trusted' alias (multiple independent users, same bar as
    get_trusted_aliases_for_prompt - no double standard between "good
    enough to bias Gemini's prompt" and "good enough to resolve a
    correction") that ALSO has a resolved food_id attached.

    This does NOT invent new nutrition data from user corrections - a
    food_id only ever gets attached to an alias row in the first place
    when record_alias_correction() was called with one (i.e. some past
    submission's canonical_name text - possibly a different spelling
    variant of the same request - already matched the static glossary
    directly). This just lets that past resolution be reused for a
    later, differently-spelled request for the same food, instead of
    re-deriving it, and it's still the static glossary's own nutrition
    numbers doing the actual work either way, never a user-supplied
    number.

    Reusing this is the "cheapest appropriate existing knowledge
    mechanism" step: free, deterministic, zero API calls, before ever
    considering a real fallback (a fallback which - by deliberate
    decision - this project still doesn't call, since correcting a
    single word to send a Gemini request would work against the exact
    RPM-pressure goal this whole correction system exists to serve; see
    _build_corrected_food's docstring).
    """
    name = _normalize_alias_text(canonical_name)
    if not name:
        return None
    conn = _connect()
    row = conn.execute(
        "SELECT food_id FROM food_aliases WHERE canonical_name = ? "
        "AND status = 'trusted' AND food_id IS NOT NULL "
        "ORDER BY last_seen_at DESC LIMIT 1",
        (name,),
    ).fetchone()
    conn.close()
    return row["food_id"] if row else None


_alias_cleanup_op_count = 0


def cleanup_stale_alias_candidates():
    """
    Deletes 'candidate' aliases (never got a second independent voter)
    older than CANDIDATE_EXPIRY_DAYS - keeps the table from growing
    forever with one-off/unconfirmed corrections, per the explicit "avoid
    storing every random correction forever" / "allow low-confidence
    candidates to expire" requirement. Never touches 'supported' or
    'trusted' rows - only genuinely unconfirmed, aging candidates.
    Called opportunistically (see record_alias_correction's caller in
    bot.py), not on every write, to keep this cheap on a 128 MB server.
    """
    try:
        cutoff = (datetime.utcnow() - timedelta(days=CANDIDATE_EXPIRY_DAYS)).isoformat()
        conn = _connect()
        conn.execute(
            "DELETE FROM food_aliases WHERE status = 'candidate' AND last_seen_at < ?",
            (cutoff,),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # cleanup is best-effort, never worth breaking anything over

