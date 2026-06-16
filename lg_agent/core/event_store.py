"""Persist a session's event stream + state snapshots to SQLite — a "tee".

The session yields one ordered stream of UI events per turn; this sink durably
records every one of them. Because the orchestrator's stream already carries the
sub-agents' traffic (their tool calls, step results, and the deep-trace frames are
re-pumped up into it), teeing that single stream captures BOTH the main agent and
every sub-agent — no per-agent wiring needed.

Two tables:
  * ``events``    — every event verbatim (one row each), JSON in ``data``.
  * ``snapshots`` — a denormalized row per ``state`` event (the cart + full
    snapshot), so "what was the state at turn N" is a trivial query.

Storage is decoupled from the session: the session takes an optional store and
calls :meth:`record_turn` once per turn (batched in one transaction). No store
wired → nothing is written (tests stay clean). SQLite is the backend for
simplicity; the same shape ports to Postgres by swapping the connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT,
    created_at TEXT,
    last_seen  TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id    TEXT,
    turn       INTEGER,
    seq        INTEGER,
    type       TEXT,
    ts         TEXT,
    data       TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn       INTEGER,
    ts         TEXT,
    cart       TEXT,
    snapshot   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, turn, seq);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id, turn);
"""


def now_iso() -> str:
    """An ISO-8601 UTC timestamp (captured at event time by the session wrapper)."""
    return datetime.now(UTC).isoformat()


class SQLiteEventStore:
    """A thread-safe SQLite sink for the session's event + snapshot stream."""

    def __init__(self, db_path: str = "lg_agent_events.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_turn(
        self, session_id: str, user_id: str, turn: int, rows: list[tuple[str, dict]]
    ) -> None:
        """Persist one turn's events (and any state snapshots) in a single transaction.

        ``rows`` is ``[(ts_iso, event_dict), ...]`` in stream order — the session
        stamps each event's time as it is yielded.
        """
        now = now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, user_id, created_at, last_seen) VALUES (?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET last_seen=excluded.last_seen",
                (session_id, user_id, now, now),
            )
            for seq, (ts, ev) in enumerate(rows):
                etype = ev.get("type", "?") if isinstance(ev, dict) else "?"
                self._conn.execute(
                    "INSERT INTO events(session_id, user_id, turn, seq, type, ts, data) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (session_id, user_id, turn, seq, etype, ts, json.dumps(ev, default=str)),
                )
                if etype == "state" and isinstance(ev.get("snapshot"), dict):
                    snap = ev["snapshot"]
                    self._conn.execute(
                        "INSERT INTO snapshots(session_id, turn, ts, cart, snapshot) "
                        "VALUES (?,?,?,?,?)",
                        (
                            session_id,
                            turn,
                            ts,
                            json.dumps(snap.get("cart"), default=str),
                            json.dumps(snap, default=str),
                        ),
                    )
            self._conn.commit()

    # ----- read helpers (verification / inspection / replay) -----
    def list_sessions(self) -> list[dict]:
        """All known sessions, newest activity first, with event + turn counts."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT s.session_id, s.user_id, s.created_at, s.last_seen, "
                "  (SELECT COUNT(*) FROM events e WHERE e.session_id = s.session_id) AS events, "
                "  (SELECT COALESCE(MAX(turn), 0) FROM events e WHERE e.session_id = s.session_id) AS turns "
                "FROM sessions s ORDER BY s.last_seen DESC"
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def read_events(self, session_id: str | None = None) -> list[dict]:
        sql = "SELECT session_id, turn, seq, type, ts, data FROM events"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY id"
        with self._lock:
            cur = self._conn.execute(sql, params)
            cols = [c[0] for c in cur.description]
            out = []
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                rec["data"] = json.loads(rec["data"])
                out.append(rec)
        return out

    def read_snapshots(self, session_id: str | None = None) -> list[dict]:
        sql = "SELECT session_id, turn, ts, cart, snapshot FROM snapshots"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY id"
        with self._lock:
            cur = self._conn.execute(sql, params)
            cols = [c[0] for c in cur.description]
            out = []
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                rec["cart"] = json.loads(rec["cart"]) if rec["cart"] else None
                rec["snapshot"] = json.loads(rec["snapshot"]) if rec["snapshot"] else None
                out.append(rec)
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["SQLiteEventStore", "now_iso"]
