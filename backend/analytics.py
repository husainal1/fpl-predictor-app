"""
Lightweight analytics + one A/B test, backed by SQLite (stdlib only).

The A/B test: every session is deterministically bucketed into "concise" or
"detailed" explanation style. We log which explanations get a thumbs-up and
expose the helpful-rate per variant at /api/stats, so there's a real,
measurable experiment (ingredient 4 of the build plan).
"""
import os
import json
import time
import sqlite3
import hashlib
from contextlib import contextmanager

DB_PATH = os.environ.get("FPL_DB_PATH", "/tmp/fpl_analytics.db")
VARIANTS = ("concise", "detailed")


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, session_id TEXT, event_type TEXT,
                player_id INTEGER, variant TEXT, meta TEXT
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, session_id TEXT, player_id INTEGER,
                variant TEXT, helpful INTEGER
            )""")


def variant_for(session_id: str) -> str:
    """Deterministic 50/50 bucket from the session id."""
    h = int(hashlib.sha256((session_id or "anon").encode()).hexdigest(), 16)
    return VARIANTS[h % 2]


def log_event(session_id, event_type, player_id=None, variant=None, meta=None):
    with _conn() as con:
        con.execute(
            "INSERT INTO events (ts, session_id, event_type, player_id, variant, meta) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), session_id, event_type, player_id, variant,
             json.dumps(meta) if meta else None),
        )


def log_feedback(session_id, player_id, variant, helpful: bool):
    with _conn() as con:
        con.execute(
            "INSERT INTO feedback (ts, session_id, player_id, variant, helpful) VALUES (?,?,?,?,?)",
            (time.time(), session_id, player_id, variant, 1 if helpful else 0),
        )


def stats():
    with _conn() as con:
        total_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        sessions = con.execute("SELECT COUNT(DISTINCT session_id) FROM events").fetchone()[0]
        by_type = dict(con.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall())

        ab = {}
        for v in VARIANTS:
            shown = con.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='explain' AND variant=?",
                (v,)).fetchone()[0]
            up = con.execute(
                "SELECT COUNT(*) FROM feedback WHERE variant=? AND helpful=1", (v,)).fetchone()[0]
            votes = con.execute(
                "SELECT COUNT(*) FROM feedback WHERE variant=?", (v,)).fetchone()[0]
            ab[v] = {
                "explanations_shown": shown,
                "thumbs_up": up,
                "votes": votes,
                "helpful_rate": round(up / votes, 3) if votes else None,
            }

        top_players = con.execute(
            "SELECT player_id, COUNT(*) c FROM events WHERE event_type='explain' "
            "GROUP BY player_id ORDER BY c DESC LIMIT 10").fetchall()

    return {
        "total_events": total_events,
        "sessions": sessions,
        "events_by_type": by_type,
        "ab_test": {"experiment": "explanation_style", "variants": ab},
        "top_explained_players": [{"player_id": p, "count": c} for p, c in top_players],
    }
