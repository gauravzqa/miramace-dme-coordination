"""B1 State — SQLite event store: append/load events, task outbox reconcile, transcripts, caches.

- `events` is append-only: no method UPDATEs or DELETEs an event row.
- append() validates the payload, assigns seq = last + 1 per case, runs the reducer and rejects (raises,
  nothing written) on IllegalTransition.
- `tasks` is the outbox (UNIQUE idem_key). `commitments` / `escalations` are rebuildable caches.
- Datetimes are stored as ISO strings; task due_at is stored in UTC so lexical order == time order.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from miramace.events import PAYLOADS, CaseOpened, Event, EventType, Source
from miramace.models import CHICAGO, Address, CallResult, CaseState, Supplier, Task, TaskSpec, TaskStatus
from miramace.reducer import IllegalTransition, initial_state, reduce, reduce_all

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY, scenario TEXT, intake_json TEXT NOT NULL,
  created_sim TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, seq INTEGER NOT NULL,
  type TEXT NOT NULL, payload_json TEXT NOT NULL, source_kind TEXT NOT NULL, source_ref TEXT NOT NULL,
  source_span TEXT, causation_id TEXT, sim_time TEXT NOT NULL, recorded_at TEXT NOT NULL,
  UNIQUE(case_id, seq));
CREATE INDEX IF NOT EXISTS ix_events_case_type ON events(case_id, type);
CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT, party_id TEXT,
  intent TEXT, version INT, idem_key TEXT NOT NULL UNIQUE, due_at TEXT NOT NULL, attempts INT DEFAULT 0,
  max_attempts INT, status TEXT, args_json TEXT, last_result TEXT);
CREATE INDEX IF NOT EXISTS ix_tasks_due ON tasks(case_id, status, due_at);
CREATE TABLE IF NOT EXISTS transcripts(transcript_id TEXT PRIMARY KEY, case_id TEXT, task_id TEXT,
  party_id TEXT, outcome TEXT, turns_json TEXT, claims_json TEXT, flags_json TEXT, transport TEXT,
  sim_time TEXT, result_json TEXT);
CREATE TABLE IF NOT EXISTS suppliers(sid TEXT PRIMARY KEY, name TEXT, phone TEXT, address_json TEXT,
  zip TEXT);
CREATE TABLE IF NOT EXISTS commitments(cid TEXT PRIMARY KEY, case_id TEXT, party TEXT, what TEXT,
  due_at TEXT, status TEXT, source_event_id TEXT);
CREATE TABLE IF NOT EXISTS escalations(eid TEXT PRIMARY KEY, case_id TEXT, lane TEXT, trigger TEXT,
  track TEXT, status TEXT, body_json TEXT, resolution TEXT);
CREATE INDEX IF NOT EXISTS ix_esc_open ON escalations(status, lane);
"""


class TranscriptRecord(BaseModel):
    """A stored call transcript (full CallResult plus routing metadata)."""

    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    case_id: str
    task_id: str
    party_id: str
    sim_time: datetime
    result: CallResult


def _iso(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError("datetime must be tz-aware")
    return t.isoformat()


def _utc(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError("datetime must be tz-aware")
    return t.astimezone(UTC).isoformat()


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(CHICAGO)


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            # Single-process tool: classic rollback journal works on every filesystem (WAL fails on
            # some shared/network mounts with "disk I/O error"). Opt in with MIRA_SQLITE_WAL=1.
            # TRUNCATE (not DELETE) so commits never unlink the journal file — works in folders where
            # deletion is restricted (sandboxed/shared mounts).
            mode = "WAL" if os.getenv("MIRA_SQLITE_WAL") == "1" else "TRUNCATE"
            self._db.execute(f"PRAGMA journal_mode={mode}")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._states: dict[str, CaseState] = {}

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ cases
    def create_case(self, case_id: str, scenario: str, intake: dict[str, Any], created_sim: datetime) -> None:
        try:
            self._db.execute(
                "INSERT INTO cases(case_id, scenario, intake_json, created_sim) VALUES (?,?,?,?)",
                (case_id, scenario, json.dumps(intake, default=str), _iso(created_sim)),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"case {case_id!r} already exists") from exc

    def case(self, case_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            return None
        return {
            "case_id": row["case_id"],
            "scenario": row["scenario"],
            "intake": json.loads(row["intake_json"]),
            "created_sim": _dt(row["created_sim"]),
        }

    def case_ids(self) -> list[str]:
        rows = self._db.execute(
            "SELECT case_id FROM cases UNION SELECT DISTINCT case_id FROM events ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ events
    def append(self, event: Event) -> Event:
        return self.append_many([event])[0]

    def append_many(self, events: list[Event]) -> list[Event]:
        """Atomically append events (all or nothing). Raises IllegalTransition / ValidationError."""
        if not events:
            return []
        work: dict[str, CaseState | None] = {}
        next_seq: dict[str, int] = {}
        stored: list[Event] = []
        for ev in events:
            PAYLOADS[EventType(ev.type)].model_validate(ev.payload)  # raises ValidationError
            cid = ev.case_id
            if cid not in work:
                work[cid] = self._current_state(cid)
                next_seq[cid] = self._last_seq(cid) + 1
            st = work[cid]
            if st is None:
                if ev.type != EventType.CASE_OPENED:
                    raise IllegalTransition(ev.type, f"case {cid!r} has no CASE_OPENED yet")
                p = ev.parsed()
                assert isinstance(p, CaseOpened)
                st = initial_state(cid, p.patient, p.pcp)
            e2 = ev.model_copy(update={"seq": next_seq[cid]})
            work[cid] = reduce(st, e2)  # raises IllegalTransition -> nothing written
            next_seq[cid] += 1
            stored.append(e2)
        recorded = datetime.now(UTC).isoformat()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.executemany(
                "INSERT INTO events(event_id, case_id, seq, type, payload_json, source_kind, source_ref,"
                " source_span, causation_id, sim_time, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        e.event_id,
                        e.case_id,
                        e.seq,
                        str(e.type),
                        json.dumps(e.payload),
                        e.source.kind,
                        e.source.ref,
                        json.dumps(list(e.source.span)) if e.source.span is not None else None,
                        e.causation_id,
                        _iso(e.sim_time),
                        recorded,
                    )
                    for e in stored
                ],
            )
            self._db.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            self._db.execute("ROLLBACK")
            raise ValueError(f"event rejected by store: {exc}") from exc
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        for cid, st in work.items():
            if st is not None:
                self._states[cid] = st
        return [e.model_copy(deep=True) for e in stored]

    def events(self, case_id: str) -> list[Event]:
        rows = self._db.execute("SELECT * FROM events WHERE case_id=? ORDER BY seq", (case_id,)).fetchall()
        return [self._row_event(r) for r in rows]

    def state(self, case_id: str) -> CaseState:
        st = self._current_state(case_id)
        if st is None:
            raise KeyError(f"no events for case {case_id!r}")
        return st.model_copy(deep=True)

    def _current_state(self, case_id: str) -> CaseState | None:
        st = self._states.get(case_id)
        if st is None:
            evs = self.events(case_id)
            if not evs:
                return None
            st = reduce_all(evs)
            self._states[case_id] = st
        return st

    def _last_seq(self, case_id: str) -> int:
        row = self._db.execute("SELECT MAX(seq) FROM events WHERE case_id=?", (case_id,)).fetchone()
        return row[0] or 0

    @staticmethod
    def _row_event(r: sqlite3.Row) -> Event:
        span = json.loads(r["source_span"]) if r["source_span"] else None
        return Event(
            event_id=r["event_id"],
            case_id=r["case_id"],
            seq=r["seq"],
            type=EventType(r["type"]),
            payload=json.loads(r["payload_json"]),
            source=Source(kind=r["source_kind"], ref=r["source_ref"], span=tuple(span) if span else None),
            causation_id=r["causation_id"],
            sim_time=_dt(r["sim_time"]),
        )

    # ------------------------------------------------------------------ tasks (outbox)
    def reconcile_tasks(self, case_id: str, specs: list[TaskSpec], now: datetime) -> list[Task]:
        """Insert specs with new idem_keys; cancel PENDING tasks no longer desired; never touch DONE/FAILED.

        An existing PENDING task keeps its original due_at (the planner may recompute due relative to `now`).
        A CANCELLED task whose idem_key is desired again is revived to PENDING with the spec's due_at.
        Returns current PENDING tasks of the case ordered by due_at.
        """
        del now  # due shifting into allowed hours is the planner's job
        wanted = {s.idem_key: s for s in specs}
        self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = (
                {
                    r["idem_key"]: r
                    for r in self._db.execute(
                        "SELECT idem_key, status, case_id FROM tasks WHERE idem_key IN ("
                        + ",".join("?" * len(wanted))
                        + ")",
                        list(wanted),
                    ).fetchall()
                }
                if wanted
                else {}
            )
            for key, spec in wanted.items():
                row = existing.get(key)
                if row is None:
                    self._db.execute(
                        "INSERT INTO tasks(task_id, case_id, kind, party_id, intent, version, idem_key,"
                        " due_at, attempts, max_attempts, status, args_json, last_result)"
                        " VALUES (?,?,?,?,?,?,?,?,0,?,?,?,NULL)",
                        (
                            uuid.uuid4().hex,
                            case_id,
                            str(spec.kind),
                            spec.party_id,
                            spec.intent,
                            spec.version,
                            key,
                            _utc(spec.due_at),
                            spec.max_attempts,
                            str(TaskStatus.PENDING),
                            json.dumps(spec.args, default=str),
                        ),
                    )
                elif row["status"] == TaskStatus.CANCELLED and row["case_id"] == case_id:
                    self._db.execute(
                        "UPDATE tasks SET status=?, due_at=?, args_json=? WHERE idem_key=?",
                        (str(TaskStatus.PENDING), _utc(spec.due_at), json.dumps(spec.args, default=str), key),
                    )
            pending = self._db.execute(
                "SELECT idem_key FROM tasks WHERE case_id=? AND status=?", (case_id, str(TaskStatus.PENDING))
            ).fetchall()
            drop = [r["idem_key"] for r in pending if r["idem_key"] not in wanted]
            self._db.executemany(
                "UPDATE tasks SET status=? WHERE idem_key=? AND status=?",
                [(str(TaskStatus.CANCELLED), k, str(TaskStatus.PENDING)) for k in drop],
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return self.tasks(case_id, TaskStatus.PENDING)

    def next_pending_due(self, case_id: str) -> Task | None:
        row = self._db.execute(
            "SELECT * FROM tasks WHERE case_id=? AND status=? ORDER BY due_at, rowid LIMIT 1",
            (case_id, str(TaskStatus.PENDING)),
        ).fetchone()
        return self._row_task(row) if row else None

    def tasks(self, case_id: str, status: TaskStatus | str | None = None) -> list[Task]:
        if status is None:
            rows = self._db.execute(
                "SELECT * FROM tasks WHERE case_id=? ORDER BY due_at, rowid", (case_id,)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM tasks WHERE case_id=? AND status=? ORDER BY due_at, rowid",
                (case_id, str(TaskStatus(status))),
            ).fetchall()
        return [self._row_task(r) for r in rows]

    def task(self, task_id: str) -> Task:
        row = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row_task(row)

    def task_last_result(self, task_id: str) -> str | None:
        row = self._db.execute("SELECT last_result FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row[0]

    def mark_task(
        self,
        task_id: str,
        status: TaskStatus | str,
        attempts: int | None = None,
        last_result: str | None = None,
    ) -> Task:
        cur = self._db.execute(
            "UPDATE tasks SET status=?, attempts=COALESCE(?, attempts), last_result=COALESCE(?, last_result)"
            " WHERE task_id=?",
            (str(TaskStatus(status)), attempts, last_result, task_id),
        )
        if cur.rowcount == 0:
            raise KeyError(task_id)
        return self.task(task_id)

    @staticmethod
    def _row_task(r: sqlite3.Row) -> Task:
        return Task(
            task_id=r["task_id"],
            case_id=r["case_id"],
            kind=r["kind"],
            party_id=r["party_id"],
            intent=r["intent"],
            version=r["version"],
            idem_key=r["idem_key"],
            due_at=_dt(r["due_at"]),
            attempts=r["attempts"] or 0,
            max_attempts=r["max_attempts"],
            status=r["status"],
            args=json.loads(r["args_json"]) if r["args_json"] else {},
        )

    # ------------------------------------------------------------------ transcripts
    def save_transcript(
        self, case_id: str, task_id: str, party_id: str, result: CallResult, sim_time: datetime
    ) -> str:
        tid = f"tr-{uuid.uuid4().hex[:16]}"
        self._db.execute(
            "INSERT INTO transcripts(transcript_id, case_id, task_id, party_id, outcome, turns_json,"
            " claims_json, flags_json, transport, sim_time, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                tid,
                case_id,
                task_id,
                party_id,
                result.outcome,
                json.dumps([u.model_dump(mode="json") for u in result.transcript]),
                json.dumps([c.model_dump(mode="json") for c in result.claims]),
                json.dumps(result.flags),
                result.transport,
                _iso(sim_time),
                result.model_dump_json(),
            ),
        )
        return tid

    def transcript(self, transcript_id: str) -> TranscriptRecord | None:
        row = self._db.execute("SELECT * FROM transcripts WHERE transcript_id=?", (transcript_id,)).fetchone()
        return self._row_transcript(row) if row else None

    def transcripts(self, case_id: str) -> list[TranscriptRecord]:
        rows = self._db.execute(
            "SELECT * FROM transcripts WHERE case_id=? ORDER BY sim_time, rowid", (case_id,)
        ).fetchall()
        return [self._row_transcript(r) for r in rows]

    def transcript_ids(self, case_id: str) -> set[str]:
        rows = self._db.execute(
            "SELECT transcript_id FROM transcripts WHERE case_id=?", (case_id,)
        ).fetchall()
        return {r[0] for r in rows}

    @staticmethod
    def _row_transcript(r: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            transcript_id=r["transcript_id"],
            case_id=r["case_id"],
            task_id=r["task_id"],
            party_id=r["party_id"],
            sim_time=_dt(r["sim_time"]),
            result=CallResult.model_validate_json(r["result_json"]),
        )

    # ------------------------------------------------------------------ suppliers
    def upsert_suppliers(self, suppliers: list[Supplier]) -> None:
        self._db.executemany(
            "INSERT INTO suppliers(sid, name, phone, address_json, zip) VALUES (?,?,?,?,?)"
            " ON CONFLICT(sid) DO UPDATE SET name=excluded.name, phone=excluded.phone,"
            " address_json=excluded.address_json, zip=excluded.zip",
            [(s.sid, s.name, s.phone, s.address.model_dump_json(), s.zip) for s in suppliers],
        )

    def suppliers(self) -> list[Supplier]:
        rows = self._db.execute("SELECT * FROM suppliers ORDER BY sid").fetchall()
        return [
            Supplier(
                sid=r["sid"],
                name=r["name"],
                phone=r["phone"],
                address=Address.model_validate_json(r["address_json"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ projection caches
    def sync_projections(self, case_id: str, state: CaseState) -> None:
        """Rewrite the commitments/escalations cache tables for the case (core never reads them)."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("DELETE FROM commitments WHERE case_id=?", (case_id,))
            self._db.execute("DELETE FROM escalations WHERE case_id=?", (case_id,))
            self._db.executemany(
                "INSERT OR REPLACE INTO commitments(cid, case_id, party, what, due_at, status,"
                " source_event_id) VALUES (?,?,?,?,?,?,?)",
                [
                    (c.cid, case_id, c.party, c.what, _iso(c.due_at), str(c.status), c.source_event_id)
                    for c in state.commitments.values()
                ],
            )
            self._db.executemany(
                "INSERT OR REPLACE INTO escalations(eid, case_id, lane, trigger, track, status, body_json,"
                " resolution) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        x.eid,
                        case_id,
                        str(x.lane),
                        str(x.trigger),
                        str(x.track),
                        str(x.status),
                        x.model_dump_json(),
                        x.resolution,
                    )
                    for x in state.escalations.values()
                ],
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def cached_escalations(self, status: str | None = None) -> list[dict[str, Any]]:
        """Read the escalations cache across cases (CLI convenience; core never reads it)."""
        q, args = "SELECT case_id, body_json FROM escalations", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        return [
            {"case_id": r["case_id"], **json.loads(r["body_json"])}
            for r in self._db.execute(q, args).fetchall()
        ]
