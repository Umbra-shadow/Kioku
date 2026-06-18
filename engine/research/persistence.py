# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""Durable history — the database behind the Researcher so nothing is ever lost.

Memory in Kioku is **per user, not per session**. One person's whole research
life — every run, every finding, every chat — accumulates into a single memory
and is written here, so:

  • a restart (or a redeploy) loses nothing — the in-RAM index is rebuilt from
    these rows;
  • a brand-new session, even years later, recalls a run from long ago;
  • the history of runs and conversations is queryable.

**Two backends, one interface.** If ``DATABASE_URL`` (a ``postgres://`` /
``postgresql://`` DSN — e.g. Neon) is provided, the durable store is cloud
Postgres, so history survives across machines and deployments. Otherwise it falls
back to a local SQLite file — zero-config for the demo. The SQL is written once in
a backend-neutral dialect; only the placeholder, the id type, and the blob type
differ.

Three tables:
  runs     — one row per investigation (topic, status, questions, findings, report)
  engrams  — the durable memory itself: every committed engram's bytes, by tenant
  chats    — the follow-up conversation, by session, so chat history survives too

One connection, guarded by a lock (the engine is single-node); writes are tiny.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from engine.engram import Engram

log = logging.getLogger("kioku.research.db")


def _is_pg(dsn: str) -> bool:
    return dsn.startswith("postgres://") or dsn.startswith("postgresql://")


def _schema(pg: bool) -> list[str]:
    serial = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    blob = "BYTEA" if pg else "BLOB"
    return [
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, topic TEXT NOT NULL,
            status TEXT NOT NULL, num_questions INTEGER, questions TEXT, findings TEXT,
            report TEXT, provider TEXT, grounded_count INTEGER, session_id TEXT,
            error TEXT, created_ts DOUBLE PRECISION, done_ts DOUBLE PRECISION)""".replace(
            "DOUBLE PRECISION", "DOUBLE PRECISION" if pg else "REAL"
        ),
        f"""CREATE TABLE IF NOT EXISTS engrams (
            tenant TEXT NOT NULL, engram_id TEXT NOT NULL,
            ts {"DOUBLE PRECISION" if pg else "REAL"}, data {blob} NOT NULL,
            PRIMARY KEY (tenant, engram_id))""",
        f"""CREATE TABLE IF NOT EXISTS chats (
            id {serial}, tenant TEXT NOT NULL, run_id TEXT, session_id TEXT,
            role TEXT NOT NULL, content TEXT NOT NULL, ts {"DOUBLE PRECISION" if pg else "REAL"})""",
        "CREATE INDEX IF NOT EXISTS ix_engrams_tenant ON engrams(tenant)",
        "CREATE INDEX IF NOT EXISTS ix_chats_session ON chats(session_id)",
        "CREATE INDEX IF NOT EXISTS ix_runs_created ON runs(created_ts)",
    ]


class ResearchDB:
    """The durable store. ``dsn`` is a Postgres URL, a SQLite path, or ``:memory:``."""

    def __init__(self, dsn: str | Path) -> None:
        self.dsn = str(dsn)
        self.pg = _is_pg(self.dsn)
        self.ph = "%s" if self.pg else "?"
        self._lock = threading.Lock()
        if self.pg:
            import psycopg2  # lazy: only needed when a Postgres DSN is used

            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
            self.backend = "postgres"
        else:
            if self.dsn != ":memory:":
                Path(self.dsn).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.dsn, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self.backend = "sqlite"
        with self._lock:
            cur = self._cursor()
            for stmt in _schema(self.pg):
                cur.execute(stmt)
            self._conn.commit()
            cur.close()
        log.info("research store ready: %s", self.backend)

    # -- backend plumbing -------------------------------------------------

    def _ensure_pg_conn(self) -> None:
        """Reconnect to Postgres if the connection was closed by the server
        (Neon and other serverless providers close idle connections)."""
        import psycopg2
        try:
            self._conn.cursor().execute("SELECT 1")
        except psycopg2.InterfaceError:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
        except psycopg2.OperationalError:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False

    def _cursor(self):
        if self.pg:
            from psycopg2.extras import RealDictCursor
            self._ensure_pg_conn()
            return self._conn.cursor(cursor_factory=RealDictCursor)
        return self._conn.cursor()

    def _q(self, sql: str) -> str:
        return sql.replace("?", self.ph) if self.pg else sql

    def _binary(self, data: bytes):
        if self.pg:
            import psycopg2

            return psycopg2.Binary(data)
        return data

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            cur = self._cursor()
            try:
                cur.execute(self._q(sql), params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def _read(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._cursor()
            try:
                cur.execute(self._q(sql), params)
                rows = cur.fetchall()
            finally:
                cur.close()
        return [dict(r) for r in rows]

    # -- runs -------------------------------------------------------------

    def save_run(self, run: dict[str, Any]) -> None:
        self._write(
            """INSERT INTO runs
               (run_id, tenant, topic, status, num_questions, questions, findings,
                report, provider, grounded_count, session_id, error, created_ts, done_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (run_id) DO UPDATE SET
                status=EXCLUDED.status, questions=EXCLUDED.questions,
                findings=EXCLUDED.findings, report=EXCLUDED.report,
                provider=EXCLUDED.provider, grounded_count=EXCLUDED.grounded_count,
                session_id=EXCLUDED.session_id, error=EXCLUDED.error, done_ts=EXCLUDED.done_ts""",
            (
                run["run_id"], run.get("tenant", ""), run["topic"], run["status"],
                run.get("num_questions"), json.dumps(run.get("questions") or []),
                json.dumps(run.get("findings") or []), run.get("report") or "",
                run.get("provider") or "", run.get("grounded_count") or 0,
                run.get("session_id") or "", run.get("error"),
                run.get("created_ts") or time.time(), run.get("done_ts"),
            ),
        )

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return _run_row(rows[0]) if rows else None

    def load_runs(self, tenant: str | None = None) -> list[dict[str, Any]]:
        if tenant:
            rows = self._read("SELECT * FROM runs WHERE tenant=? ORDER BY created_ts DESC", (tenant,))
        else:
            rows = self._read("SELECT * FROM runs ORDER BY created_ts DESC")
        return [_run_row(r) for r in rows]

    # -- engrams (the durable memory) ------------------------------------

    def save_engram(self, tenant: str, engram: Engram) -> None:
        self._write(
            """INSERT INTO engrams (tenant, engram_id, ts, data) VALUES (?,?,?,?)
               ON CONFLICT (tenant, engram_id) DO UPDATE SET ts=EXCLUDED.ts, data=EXCLUDED.data""",
            (tenant, engram.engram_id, engram.ts, self._binary(engram.to_bytes())),
        )

    def load_engrams(self, tenant: str) -> Iterator[Engram]:
        rows = self._read("SELECT data FROM engrams WHERE tenant=? ORDER BY ts ASC", (tenant,))
        for r in rows:
            try:
                yield Engram.from_bytes(bytes(r["data"]))
            except Exception:  # noqa: BLE001 — one corrupt row must not block recall
                log.warning("skipping unreadable engram row for %s", tenant)

    def count_engrams(self, tenant: str) -> int:
        rows = self._read("SELECT COUNT(*) AS n FROM engrams WHERE tenant=?", (tenant,))
        return int(rows[0]["n"]) if rows else 0

    # -- chats ------------------------------------------------------------

    def save_chat(self, tenant: str, run_id: str, session_id: str, role: str, content: str) -> None:
        self._write(
            "INSERT INTO chats (tenant, run_id, session_id, role, content, ts) VALUES (?,?,?,?,?,?)",
            (tenant, run_id, session_id, role, content, time.time()),
        )

    def load_chats(self, *, session_id: str | None = None, run_id: str | None = None,
                   tenant: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        clauses, params = [], []
        if session_id:
            clauses.append("session_id=?"); params.append(session_id)
        if run_id:
            clauses.append("run_id=?"); params.append(run_id)
        if tenant:
            clauses.append("tenant=?"); params.append(tenant)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._read(
            f"SELECT run_id, session_id, role, content, ts FROM chats{where} ORDER BY id ASC LIMIT ?",
            (*params, limit),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _run_row(row: dict) -> dict[str, Any]:
    return {
        "run_id": row["run_id"], "tenant": row["tenant"], "topic": row["topic"],
        "status": row["status"], "num_questions": row["num_questions"],
        "questions": json.loads(row["questions"] or "[]"),
        "findings": json.loads(row["findings"] or "[]"),
        "report": row["report"] or "", "provider": row["provider"] or "",
        "grounded_count": row["grounded_count"] or 0, "session_id": row["session_id"] or "",
        "error": row["error"], "created_ts": row["created_ts"], "done_ts": row["done_ts"],
    }
