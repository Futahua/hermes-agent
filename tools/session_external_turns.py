#!/usr/bin/env python3
"""A durable inbox for one non-interrupting activation of a stored session.

WHAT THIS IS FOR

A local process that is not Hermes finishes some work and needs to tell the
person about it, in the conversation they already have open. Every existing
notification rail assumes the producer runs INSIDE the process that owns that
live session -- async delegations are children of the turn that dispatched
them, kanban rows are addressed to a subscription this gateway registered. An
unrelated local process had no way in at all, and its only recourse was to open
a SECOND owner of the session and write there, which is precisely what the
active-session lease refuses.

So the producer does not deliver. It enqueues, and whichever process
legitimately owns the session consumes.

WHY THAT ORDERING MATTERS

The obvious design asks "does this session have a live owner?" and picks a
transport from the answer. That answer is stale the moment it is read: an owner
can appear or die in the gap before delivery, and one of the two branches is
then wrong in a way that either loses the event or writes it twice.

Enqueueing first removes the branch. There is one durable event and a rule about
who may take it, and the active-session lease -- not a preflight guess -- decides
that at the moment of consumption:

    A owns S   -> re-enters its own lease -> claims the event
    B does not -> SESSION_NOT_OWNED       -> leaves it alone

The race still happens; there is no longer an unsafe outcome of it.

THE LIFECYCLE IS ABOUT THE TURN, NOT ONLY ABOUT INGRESS

    PENDING -> CLAIMED -> STARTED -> FINISHED

A producer reading canonical history has to tell two situations apart that look
identical in the transcript: its marker is present with no assistant reply
because the turn is still being reasoned about, and its marker is present with
no assistant reply because the turn died. Under a direct submit those were
distinguishable, because the submitting process's own liveness answered it.
Under this rail the turn is hosted by somebody else entirely, so the inbox has to
say so:

    STARTED  + owner live   -> still going; do not judge it yet
    STARTED  + owner dead   -> reconcile against history; never guess
    FINISHED                -> the turn ended, and history is now complete

Without that distinction a healthy long turn reads as a partial delivery.

EVERY TRANSITION IS BOUND TO A CLAIM

``claim_id`` is minted per successful claim and every later mutation is
compare-and-swapped against it. Two processes that both saw the same dead claim
can both try to recover it; only one wins, and the loser's later "mark started"
or "release" cannot land on the winner's claim. The active-session lease makes
that race hard to reach through the current consumer, but this module advertises
its own mutual exclusion, so it provides it rather than borrowing an invariant
from its caller.

WHAT THIS IS NOT

It is not a second delivery ledger. The producer's own outbox remains the
delivery authority and canonical Hermes history remains the record of what was
said. ``event_id`` is the PRODUCER's identity for the event, so re-enqueueing
after an ambiguous outcome is idempotent here by construction.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PENDING = "PENDING"
CLAIMED = "CLAIMED"
STARTED = "STARTED"
FINISHED = "FINISHED"

# Advertised through the gateway. Distinct from the active-session lease
# capability on purpose: that one proves only that concurrent writers to a
# session are fenced, which a build can do without having this inbox or the
# poller that drains it. A producer that conflated them would enqueue events
# into a build where nothing would ever consume them.
SESSION_EXTERNAL_TURNS_V1 = True


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    # Mirrors hermes_state_common.SCHEMA_SQL. Repeated here so a producer that
    # never opens a full Hermes state handle still finds the table present.
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (session_external_turns)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS session_external_turns (
            event_id TEXT PRIMARY KEY,
            target_session_key TEXT NOT NULL,
            body TEXT NOT NULL,
            source TEXT NOT NULL,
            display_metadata TEXT,
            state TEXT NOT NULL DEFAULT 'PENDING',
            claim_id TEXT,
            owner_pid INTEGER,
            owner_started_at REAL,
            claimed_at REAL,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            outcome TEXT,
            last_error TEXT
        )"""
    )
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(session_external_turns)")
    }
    if "display_metadata" not in columns:
        conn.execute("ALTER TABLE session_external_turns ADD COLUMN display_metadata TEXT")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_session_external_turns_pending
           ON session_external_turns(target_session_key, state, created_at)"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Commit/rollback AND close. ``with _connect()`` alone leaks the handle."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _process_start_time(pid: int) -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _claimer_alive(pid: Any, started_at: Any) -> bool:
    """Is the process holding this row still running?

    Identity is (pid, process start time) for the same reason the active-session
    registry uses it: a pid on its own is reused, and a recycled one would keep
    an abandoned claim alive forever.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        if not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    if started_at is None:
        return True
    current = _process_start_time(pid_int)
    if current is None:
        return True
    try:
        return abs(current - float(started_at)) < 0.001
    except (TypeError, ValueError):
        return True


def enqueue_external_turn(
    *,
    event_id: str,
    target_session_key: str,
    body: str,
    source: str,
    display_metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Queue one activation for a stored session. Returns False if already queued.

    Idempotent on ``event_id``: a producer that could not tell whether its last
    attempt landed may safely enqueue the same event again, and will not create a
    second turn. Nothing here delivers -- see the module docstring for why the
    producer must not also choose the transport.
    """
    key = str(target_session_key or "").strip()
    eid = str(event_id or "").strip()
    if not eid or not key:
        raise ValueError("event_id and target_session_key are both required")
    if display_metadata is not None and not isinstance(display_metadata, Mapping):
        raise ValueError("display_metadata must be an object when provided")
    metadata_json = (
        json.dumps(dict(display_metadata), ensure_ascii=False, sort_keys=True)
        if display_metadata is not None
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO session_external_turns
               (event_id, target_session_key, body, source, display_metadata, state, created_at)
               VALUES (?, ?, ?, ?, ?, 'PENDING', ?)""",
            (eid, key, str(body), str(source or "external"), metadata_json, time.time()),
        )
        return bool(cur.rowcount)


def get_external_turn(event_id: str) -> Optional[Dict[str, Any]]:
    """The row as it stands, plus whether whoever holds it is still alive.

    ``owner_alive`` is what lets a producer read STARTED correctly: with a live
    owner the turn is still being reasoned about and must not be judged; with a
    dead one the transcript is all there is, and the producer reconciles against
    it rather than guessing from this table.
    """
    with _transaction() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM session_external_turns WHERE event_id = ?", (str(event_id),)
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    if record.get("display_metadata"):
        try:
            decoded = json.loads(record["display_metadata"])
            record["display_metadata"] = decoded if isinstance(decoded, dict) else None
        except (TypeError, ValueError):
            record["display_metadata"] = None
    record["owner_alive"] = bool(
        record.get("state") in (CLAIMED, STARTED)
        and _claimer_alive(record.get("owner_pid"), record.get("owner_started_at"))
    )
    return record


def pending_external_turns(target_session_key: str, limit: int = 16) -> List[Dict[str, Any]]:
    """Rows this session may still consume, oldest first.

    PENDING rows, and CLAIMED rows whose holder died before dispatching: a
    process killed between claiming and starting must not take the event with
    it, because the producer believes it handed the event over and will not
    re-send it.

    A dead STARTED row is deliberately NOT offered. A turn began, so the marker
    may already be in the transcript, and re-dispatching would announce one thing
    twice. Whether that turn actually said anything is a question about canonical
    history, which is the producer's to answer.
    """
    key = str(target_session_key or "").strip()
    if not key:
        return []
    rows: List[Dict[str, Any]] = []
    with _transaction() as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            """SELECT * FROM session_external_turns
               WHERE target_session_key = ? AND state IN ('PENDING', 'CLAIMED')
               ORDER BY created_at LIMIT ?""",
            (key, int(limit)),
        ):
            record = dict(row)
            if record.get("display_metadata"):
                try:
                    decoded = json.loads(record["display_metadata"])
                    record["display_metadata"] = decoded if isinstance(decoded, dict) else None
                except (TypeError, ValueError):
                    record["display_metadata"] = None
            if record.get("state") == CLAIMED and _claimer_alive(
                record.get("owner_pid"), record.get("owner_started_at")
            ):
                continue
            rows.append(record)
    return rows


def claim_external_turn(event_id: str) -> Optional[str]:
    """Take ownership of one row for THIS process; returns a claim id, or None.

    The UPDATE is the whole mutual exclusion. It matches only the exact state AND
    claim this caller just observed, so two processes recovering the same dead
    claim cannot both come away believing they own it: SQLite serialises the
    write and the loser sees rowcount 0. The returned id must be presented for
    every later transition on this row.
    """
    eid = str(event_id or "").strip()
    if not eid:
        return None
    pid = os.getpid()
    started = _process_start_time(pid)
    claim_id = uuid.uuid4().hex
    now = time.time()
    with _transaction() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT state, claim_id, owner_pid, owner_started_at "
            "FROM session_external_turns WHERE event_id = ?",
            (eid,),
        ).fetchone()
        if row is None or row["state"] in (STARTED, FINISHED):
            return None
        if row["state"] == CLAIMED and _claimer_alive(row["owner_pid"], row["owner_started_at"]):
            return None
        # Bind to the observed claim as well as the observed state: recovering a
        # dead claim must fail if somebody else recovered it first.
        prior = row["claim_id"]
        cur = conn.execute(
            """UPDATE session_external_turns
               SET state = 'CLAIMED', claim_id = ?, owner_pid = ?, owner_started_at = ?,
                   claimed_at = ?
               WHERE event_id = ? AND state = ? AND claim_id IS ?""",
            (claim_id, pid, started, now, eid, row["state"], prior),
        )
        return claim_id if cur.rowcount else None


def mark_external_turn_started(event_id: str, claim_id: str) -> bool:
    """Commit this claim to the uncertain dispatch boundary.

    This transition must be durable BEFORE ``_run_prompt_submit`` launches the
    turn thread.  Once dispatch begins, a wake marker can reach canonical
    history at any instant; leaving the row CLAIMED until afterwards would make
    a crash look like "never dispatched" and allow another process to replay a
    wake that was already recorded.

    A refused dispatch is rolled back by ``release_external_turn``.  A crash
    after this transition deliberately leaves STARTED for producer-side
    canonical-history reconciliation.
    """
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE session_external_turns
               SET state = 'STARTED', started_at = ?
               WHERE event_id = ? AND state = 'CLAIMED' AND claim_id = ?""",
            (time.time(), str(event_id), str(claim_id)),
        )
        return bool(cur.rowcount)


def mark_external_turn_finished(event_id: str, claim_id: str, outcome: str = "completed") -> bool:
    """That turn has ended, so canonical history is now complete for this event.

    This is the signal that makes "marker present, no assistant reply" mean
    something definite. Until it lands, that shape is simply a turn still in
    progress.
    """
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE session_external_turns
               SET state = 'FINISHED', finished_at = ?, outcome = ?
               WHERE event_id = ? AND state = 'STARTED' AND claim_id = ?""",
            (time.time(), str(outcome), str(event_id), str(claim_id)),
        )
        return bool(cur.rowcount)


def release_external_turn(event_id: str, claim_id: str, error: str = "") -> bool:
    """Put this process's un-dispatched claim back.

    STARTED is accepted because consumers commit that state immediately before
    attempting dispatch.  Callers must use this only when dispatch returned
    False or raised before launching; claim-id CAS prevents any other owner from
    rolling the row back.
    """
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE session_external_turns
               SET state = 'PENDING', claim_id = NULL, owner_pid = NULL,
                   owner_started_at = NULL, claimed_at = NULL, started_at = NULL,
                   last_error = ?
               WHERE event_id = ? AND state IN ('CLAIMED', 'STARTED') AND claim_id = ?""",
            (str(error)[:500] or None, str(event_id), str(claim_id)),
        )
        return bool(cur.rowcount)


def reopen_external_turn(event_id: str, reason: str = "") -> bool:
    """Make a dead STARTED event deliverable again. The PRODUCER decides this.

    There is a real window in which STARTED is durable and the marker is not:
    the consumer commits STARTED immediately before calling _run_prompt_submit,
    and the launched thread persists the user row afterwards. A process killed
    in between leaves a row saying dispatch may have started and a transcript
    containing no evidence of it.

    The inbox cannot resolve that on its own, and must not try -- deciding
    whether the event landed means reading canonical history, which is the
    producer's authority, not this table's. So the producer reconciles and then
    says so here, and only for an event whose owner is gone:

        marker present -> the turn spoke, or partly spoke; ordinary
                          reconciliation applies and this is NOT called
        marker absent  -> nothing was ever written; reopen and let a live owner
                          deliver it

    Refuses while the owner is alive, so it can never yank an event out of a turn
    that is still being reasoned about.
    """
    eid = str(event_id or "").strip()
    if not eid:
        return False
    with _transaction() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT state, claim_id, owner_pid, owner_started_at "
            "FROM session_external_turns WHERE event_id = ?",
            (eid,),
        ).fetchone()
        if row is None or row["state"] != STARTED:
            return False
        if _claimer_alive(row["owner_pid"], row["owner_started_at"]):
            return False
        cur = conn.execute(
            """UPDATE session_external_turns
               SET state = 'PENDING', claim_id = NULL, owner_pid = NULL,
                   owner_started_at = NULL, claimed_at = NULL, started_at = NULL,
                   last_error = ?
               WHERE event_id = ? AND state = 'STARTED' AND claim_id IS ?""",
            (str(reason)[:500] or "reopened after owner died", eid, row["claim_id"]),
        )
        return bool(cur.rowcount)
