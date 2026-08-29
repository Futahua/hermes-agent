"""The durable inbox that carries one external activation into a stored session.

The cross-process behaviour -- who is entitled to consume, and when -- is proved
by ``scripts/probe_external_turn_route.py``, which races two real gateways.
These cover the storage contract that probe depends on: identity, claim
generation, the turn lifecycle a producer reads, and the fact that an event is
never silently lost.
"""

import os

import pytest

from tools.session_external_turns import (
    CLAIMED,
    FINISHED,
    PENDING,
    STARTED,
    _transaction,
    claim_external_turn,
    enqueue_external_turn,
    get_external_turn,
    mark_external_turn_finished,
    mark_external_turn_started,
    pending_external_turns,
    release_external_turn,
    reopen_external_turn,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def enqueue(event_id="E1", key="S", body="done", source="delegate-wave"):
    return enqueue_external_turn(
        event_id=event_id, target_session_key=key, body=body, source=source
    )


def kill_holder(event_id, pid=0x7FFFFFFE, started=1.0):
    """Rewrite the holder as a process that cannot exist."""
    with _transaction() as conn:
        conn.execute(
            "UPDATE session_external_turns SET owner_pid = ?, owner_started_at = ? "
            "WHERE event_id = ?",
            (pid, started, event_id),
        )


# ── identity ──────────────────────────────────────────────────────────────


def test_re_enqueueing_one_event_does_not_produce_two_turns():
    """The producer may not know whether its last attempt landed.

    A wake whose outcome was ambiguous gets re-sent, and it must not become two
    announcements of one thing. ``event_id`` is the producer's identity for the
    event, so idempotence is structural rather than a de-dup heuristic.
    """
    assert enqueue() is True
    assert enqueue() is False
    assert len(pending_external_turns("S")) == 1


def test_structured_display_metadata_round_trips_and_first_event_wins():
    metadata = {
        "reason": "QUESTION",
        "delegate_session_id": "asess_1",
        "delegate_message_id": "msg_1",
    }
    assert enqueue_external_turn(
        event_id="W1",
        target_session_key="S",
        body="Which API?",
        source="delegate-wave",
        display_metadata=metadata,
    ) is True
    assert enqueue_external_turn(
        event_id="W1",
        target_session_key="S",
        body="forged replacement",
        source="delegate-wave",
        display_metadata={"reason": "COMPLETED"},
    ) is False
    row = get_external_turn("W1")
    assert row["body"] == "Which API?"
    assert row["display_metadata"] == metadata


def test_an_event_is_only_visible_to_its_target_session():
    enqueue(event_id="E1", key="S1")
    enqueue(event_id="E2", key="S2")
    assert [r["event_id"] for r in pending_external_turns("S1")] == ["E1"]
    assert [r["event_id"] for r in pending_external_turns("S2")] == ["E2"]


def test_events_are_offered_oldest_first():
    enqueue(event_id="first")
    enqueue(event_id="second")
    assert [r["event_id"] for r in pending_external_turns("S")] == ["first", "second"]


def test_an_event_must_name_both_itself_and_its_target():
    with pytest.raises(ValueError):
        enqueue_external_turn(event_id="", target_session_key="S", body="x", source="dw")
    with pytest.raises(ValueError):
        enqueue_external_turn(event_id="E", target_session_key="", body="x", source="dw")


# ── claiming ──────────────────────────────────────────────────────────────


def test_a_live_claim_hides_the_row_from_everyone_else():
    enqueue()
    assert claim_external_turn("E1")
    assert pending_external_turns("S") == []
    assert claim_external_turn("E1") is None


def test_only_one_of_two_recoverers_of_a_dead_claim_wins():
    """The race this module exists to settle on its own.

    Two processes can both observe the SAME dead claim and both try to take it
    over. Binding the update to the claim they observed means the second one
    finds the row already moved and loses cleanly -- without this the state alone
    still reads CLAIMED and both would proceed.
    """
    enqueue()
    first = claim_external_turn("E1")
    assert first
    kill_holder("E1")

    # Both contenders read the same dead claim; both attempt recovery.
    recovered = claim_external_turn("E1")
    assert recovered and recovered != first
    # The loser observed the same dead claim but is now stale.
    assert claim_external_turn("E1") is None


def test_a_stale_claim_cannot_drive_the_row_it_no_longer_holds():
    """A recovered-from claim must not be able to act afterwards.

    The dead process may not be dead -- it may be wedged, and wake up holding an
    id that no longer owns anything. Every transition is bound to the claim, so
    its late writes land nowhere.
    """
    enqueue()
    stale = claim_external_turn("E1")
    kill_holder("E1")
    live = claim_external_turn("E1")
    assert live and stale != live

    assert mark_external_turn_started("E1", stale) is False
    assert release_external_turn("E1", stale) is False
    assert mark_external_turn_started("E1", live) is True
    assert mark_external_turn_finished("E1", stale) is False
    assert mark_external_turn_finished("E1", live) is True


def test_a_released_row_becomes_available_again():
    """The owner found itself busy after claiming, so the event goes back.

    This is what keeps a busy session from swallowing an event: the claim is
    provisional until a turn actually starts.
    """
    enqueue()
    claim = claim_external_turn("E1")
    assert release_external_turn("E1", claim, "session became busy") is True
    rows = pending_external_turns("S")
    assert [r["event_id"] for r in rows] == ["E1"]
    assert rows[0]["last_error"] == "session became busy"
    assert rows[0]["claim_id"] is None
    assert claim_external_turn("E1")


def test_a_pre_dispatch_started_row_can_be_released_if_dispatch_refuses():
    """STARTED closes the crash window without swallowing a refused launch."""
    enqueue()
    claim = claim_external_turn("E1")
    assert mark_external_turn_started("E1", claim) is True

    assert release_external_turn("E1", claim, "dispatch refused") is True
    row = pending_external_turns("S")[0]
    assert row["state"] == PENDING
    assert row["started_at"] is None
    assert row["last_error"] == "dispatch refused"


def test_a_dead_claimer_does_not_take_the_event_with_it():
    """A process killed between claiming and dispatching must not strand it.

    The producer believes it handed the event over and will not re-send it, so a
    row left invisible here means the announcement simply never arrives.
    """
    enqueue()
    assert claim_external_turn("E1")
    kill_holder("E1")
    assert [r["event_id"] for r in pending_external_turns("S")] == ["E1"]
    assert claim_external_turn("E1")


def test_a_recycled_pid_does_not_look_like_a_live_claimer():
    """Identity is (pid, start time). The number alone is reused."""
    enqueue()
    assert claim_external_turn("E1")
    kill_holder("E1", pid=os.getpid(), started=1.0)  # our pid, not our start time
    assert [r["event_id"] for r in pending_external_turns("S")] == ["E1"]


# ── the turn lifecycle a producer reads ───────────────────────────────────


def test_a_started_turn_reports_whether_it_is_still_alive():
    """The distinction the whole lifecycle exists for.

    "Marker present, no assistant reply" is the same transcript whether the turn
    is mid-thought or died. With a live owner the producer must wait; with a dead
    one it reconciles against history. Reading STARTED as partial delivery would
    turn every healthy long turn into a false result.
    """
    enqueue()
    claim = claim_external_turn("E1")
    mark_external_turn_started("E1", claim)

    row = get_external_turn("E1")
    assert row["state"] == STARTED
    assert row["owner_alive"] is True, "this process is running the turn"

    kill_holder("E1")
    assert get_external_turn("E1")["owner_alive"] is False


def test_a_dead_started_turn_is_never_re_dispatched():
    """A turn began, so the marker may already be in the transcript.

    Re-offering it would announce one thing twice. Whether that turn actually
    said anything is a question about canonical history, and it belongs to the
    producer -- this table only says that a turn started and stopped being
    watched.
    """
    enqueue()
    claim = claim_external_turn("E1")
    mark_external_turn_started("E1", claim)
    kill_holder("E1")

    assert pending_external_turns("S") == []
    assert claim_external_turn("E1") is None


def test_a_dead_state_eligible_for_redelivery_cannot_be_the_dispatch_state():
    """The consumer commits STARTED before any durable wake marker is possible.

    STARTED is intentionally excluded from automatic recovery.  This pins the
    structural invariant behind crash safety: after dispatch begins, owner death
    can only hand control to producer reconciliation, never another consumer.
    """
    enqueue(body="completion\n\n[delegate-wave-wake:wake_1]")
    claim = claim_external_turn("E1")
    assert mark_external_turn_started("E1", claim) is True
    kill_holder("E1")

    assert get_external_turn("E1")["state"] == STARTED
    assert pending_external_turns("S") == []


def test_finishing_a_turn_makes_the_transcript_final():
    enqueue()
    claim = claim_external_turn("E1")
    mark_external_turn_started("E1", claim)
    assert mark_external_turn_finished("E1", claim, "completed") is True

    row = get_external_turn("E1")
    assert row["state"] == FINISHED
    assert row["outcome"] == "completed"
    assert row["finished_at"] is not None
    assert row["owner_alive"] is False, "a finished turn has no live owner to wait on"
    assert pending_external_turns("S") == []
    assert claim_external_turn("E1") is None
    assert enqueue() is False, "and it cannot be resurrected under the same id"


def test_a_turn_cannot_finish_without_having_started():
    """The states are ordered, so a producer can trust what it reads."""
    enqueue()
    claim = claim_external_turn("E1")
    assert mark_external_turn_finished("E1", claim) is False
    assert get_external_turn("E1")["state"] == CLAIMED


def test_the_row_carries_what_the_consumer_needs():
    enqueue(body="done - fixed the run filter", source="delegate-wave")
    row = pending_external_turns("S")[0]
    assert row["body"] == "done - fixed the run filter"
    assert row["source"] == "delegate-wave"
    assert row["state"] == PENDING


def test_an_unknown_event_reads_as_absent_rather_than_failing():
    assert get_external_turn("never-existed") is None


# ── recovery of the STARTED-without-marker window ─────────────────────────


def test_a_dead_started_event_can_be_reopened_by_the_producer():
    """The window is real: STARTED is durable before the marker is.

    ``_run_prompt_submit`` returns once the turn THREAD is running, and that
    thread persists the user row afterwards. A process killed in between leaves
    a row saying a turn began and a transcript with no evidence of one --
    observed at kill delays up to 0.4s by scripts/probe_external_turn_crash.py.

    Only the producer can resolve it, because only the producer reads canonical
    history, so the transition is explicit rather than inferred here.
    """
    enqueue()
    claim = claim_external_turn("E1")
    mark_external_turn_started("E1", claim)
    kill_holder("E1")

    assert reopen_external_turn("E1", "no marker in history") is True
    row = get_external_turn("E1")
    assert row["state"] == PENDING
    assert row["claim_id"] is None
    assert row["started_at"] is None, "the abandoned attempt must not look like a real one"
    assert [r["event_id"] for r in pending_external_turns("S")] == ["E1"]
    assert claim_external_turn("E1")


def test_reopening_refuses_while_the_turn_is_still_running():
    """It must never yank an event out of a turn being reasoned about."""
    enqueue()
    claim = claim_external_turn("E1")
    mark_external_turn_started("E1", claim)
    assert get_external_turn("E1")["owner_alive"] is True
    assert reopen_external_turn("E1", "impatient") is False
    assert get_external_turn("E1")["state"] == STARTED


def test_reopening_only_applies_to_started():
    """A row that never dispatched recovers on its own; a finished one is done."""
    enqueue(event_id="P1")
    assert reopen_external_turn("P1") is False, "PENDING needs no reopening"

    enqueue(event_id="C1")
    claim_c = claim_external_turn("C1")
    kill_holder("C1")
    assert reopen_external_turn("C1") is False, "a dead CLAIMED row is already offered"
    assert "C1" in [r["event_id"] for r in pending_external_turns("S")]

    enqueue(event_id="F1")
    claim_f = claim_external_turn("F1")
    mark_external_turn_started("F1", claim_f)
    mark_external_turn_finished("F1", claim_f)
    kill_holder("F1")
    assert reopen_external_turn("F1") is False, "a finished turn is not re-announced"
