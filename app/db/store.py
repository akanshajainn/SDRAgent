from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    pain_points_json TEXT NOT NULL,
    value_props_json TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    raw_excerpt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (lead_id) REFERENCES leads(id)
);

CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER NOT NULL,
    research_snapshot_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    call_to_action TEXT NOT NULL,
    reflection_rounds INTEGER NOT NULL,
    final_critique_score INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (lead_id) REFERENCES leads(id),
    FOREIGN KEY (research_snapshot_id) REFERENCES research_snapshots(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id INTEGER NOT NULL,
    relevance INTEGER NOT NULL,
    personalization INTEGER NOT NULL,
    tone INTEGER NOT NULL,
    clarity INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    overall_score REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (email_id) REFERENCES emails(id)
);
"""


class LeadStore:
    """Persistence boundary for SQLite.

    This class is intentionally the only place that knows SQL details. Other
    modules call semantic methods (persist run, fetch metrics, fetch CRM rows)
    and do not build queries directly.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize DB schema once per app process."""
        async with self._init_lock:
            if self._initialized:
                return
            await self._initialize_once()
            self._initialized = True

    async def _initialize_once(self) -> None:
        """Create parent directory and ensure required schema exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            await conn.commit()

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Idempotently apply base schema and lightweight migrations."""
        # Always run CREATE TABLE IF NOT EXISTS to recover from partially initialized DB files.
        await conn.executescript(SCHEMA_SQL)
        await self._ensure_column(
            conn,
            table="research_snapshots",
            column="raw_excerpt",
            ddl="ALTER TABLE research_snapshots ADD COLUMN raw_excerpt TEXT NOT NULL DEFAULT ''",
        )

    async def _ensure_column(
        self,
        conn: aiosqlite.Connection,
        table: str,
        column: str,
        ddl: str,
    ) -> None:
        """Add one missing column for backward-compatible migrations."""
        cursor = await conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        existing = {str(row[1]) for row in rows}
        if column not in existing:
            await conn.execute(ddl)

    async def _table_columns(self, conn: aiosqlite.Connection, table: str) -> set[str]:
        """Return currently available columns for a table name."""
        cursor = await conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        return {str(row[1]) for row in rows}

    async def persist_agent_run(  # noqa: C901
        self,
        domain: str,
        company_name: str,
        summary: str,
        pain_points: list[str],
        value_props: list[str],
        sources: list[str],
        raw_excerpt: str,
        subject: str,
        body: str,
        call_to_action: str,
        reflection_rounds: int,
        final_critique_score: int,
        evaluation: dict[str, int | str | float],
    ) -> dict[str, int]:
        """Persist one complete agent run in a single transaction.

        Why transactional:
        - If any insert fails, we roll back everything and avoid partial CRM
          state (for example: lead inserted but evaluation missing).
        """
        await self.initialize()
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            await conn.execute("BEGIN")
            try:
                await conn.execute(
                    """
                    INSERT INTO leads (domain, company_name, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET company_name = excluded.company_name
                    """,
                    (domain, company_name, now),
                )
                cursor = await conn.execute("SELECT id FROM leads WHERE domain = ?", (domain,))
                lead_row = await cursor.fetchone()
                if lead_row is None:
                    raise RuntimeError("failed to load lead id")
                lead_id = int(lead_row[0])

                snapshot_columns = await self._table_columns(conn, "research_snapshots")
                snapshot_payload: dict[str, str | int] = {
                    "lead_id": lead_id,
                    "summary": summary,
                    "pain_points_json": json.dumps(pain_points),
                    "value_props_json": json.dumps(value_props),
                    "sources_json": json.dumps(sources),
                    "raw_excerpt": raw_excerpt,
                    "created_at": now,
                }
                if "homepage_title" in snapshot_columns:
                    snapshot_payload["homepage_title"] = ""
                if "raw_notes_json" in snapshot_columns:
                    snapshot_payload["raw_notes_json"] = json.dumps(
                        {"raw_excerpt": raw_excerpt, "sources": sources}
                    )

                cols = [c for c in snapshot_payload if c in snapshot_columns]
                vals = [snapshot_payload[c] for c in cols]
                placeholders = ", ".join("?" for _ in cols)
                col_sql = ", ".join(cols)
                cursor = await conn.execute(
                    # Safe here: columns are derived from fixed schema keys, not user input.
                    f"INSERT INTO research_snapshots ({col_sql}) VALUES ({placeholders})",  # noqa: S608
                    tuple(vals),
                )
                research_snapshot_id = int(cursor.lastrowid)

                cursor = await conn.execute(
                    """
                    INSERT INTO emails (
                        lead_id, research_snapshot_id, subject, body, call_to_action,
                        reflection_rounds, final_critique_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead_id,
                        research_snapshot_id,
                        subject,
                        body,
                        call_to_action,
                        reflection_rounds,
                        final_critique_score,
                        now,
                    ),
                )
                email_id = int(cursor.lastrowid)

                await conn.execute(
                    """
                    INSERT INTO evaluations (
                        email_id, relevance, personalization, tone, clarity,
                        rationale, overall_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        email_id,
                        int(evaluation["relevance"]),
                        int(evaluation["personalization"]),
                        int(evaluation["tone"]),
                        int(evaluation["clarity"]),
                        str(evaluation["rationale"]),
                        float(evaluation["overall_score"]),
                        now,
                    ),
                )

                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        return {
            "lead_id": lead_id,
            "research_snapshot_id": research_snapshot_id,
            "email_id": email_id,
        }

    async def metrics_7d(self) -> dict[str, float | int]:
        """Return evaluation volume and average overall score for the last 7 days."""
        await self.initialize()
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            cursor = await conn.execute(
                """
                SELECT COUNT(*), AVG(overall_score)
                FROM evaluations
                WHERE created_at >= ?
                """,
                (since,),
            )
            row = await cursor.fetchone()
        total = int(row[0]) if row and row[0] is not None else 0
        avg = float(row[1]) if row and row[1] is not None else 0.0
        return {"evaluations_last_7d": total, "avg_overall_score_last_7d": round(avg, 2)}

    async def dimension_trends(self, days: int = 14) -> dict[str, object]:
        """Return dimension trend data for quality monitoring views.

        Includes:
        - `last_7d`: current rolling averages by dimension
        - `daily`: per-day averages for chart/table rendering
        """
        await self.initialize()
        safe_days = max(3, min(90, int(days)))
        now = datetime.now(timezone.utc)
        since_7d = (now - timedelta(days=7)).isoformat()
        since_days = (now - timedelta(days=safe_days - 1)).isoformat()

        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            avg_cursor = await conn.execute(
                """
                SELECT
                    AVG(relevance),
                    AVG(personalization),
                    AVG(tone),
                    AVG(clarity),
                    AVG(overall_score)
                FROM evaluations
                WHERE created_at >= ?
                """,
                (since_7d,),
            )
            avg_row = await avg_cursor.fetchone()

            daily_cursor = await conn.execute(
                """
                SELECT
                    substr(created_at, 1, 10) AS day,
                    COUNT(*) AS samples,
                    AVG(relevance),
                    AVG(personalization),
                    AVG(tone),
                    AVG(clarity),
                    AVG(overall_score)
                FROM evaluations
                WHERE created_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (since_days,),
            )
            daily_rows = await daily_cursor.fetchall()

        def _f(value: object) -> float:
            if value is None:
                return 0.0
            return round(float(value), 2)

        last_7d = {
            "relevance": _f(avg_row[0] if avg_row else None),
            "personalization": _f(avg_row[1] if avg_row else None),
            "tone": _f(avg_row[2] if avg_row else None),
            "clarity": _f(avg_row[3] if avg_row else None),
            "overall": _f(avg_row[4] if avg_row else None),
        }

        daily: list[dict[str, object]] = []
        for row in daily_rows:
            daily.append(
                {
                    "day": str(row[0]),
                    "samples": int(row[1]),
                    "relevance": _f(row[2]),
                    "personalization": _f(row[3]),
                    "tone": _f(row[4]),
                    "clarity": _f(row[5]),
                    "overall": _f(row[6]),
                }
            )
        return {"last_7d": last_7d, "daily": daily}

    async def recent_crm_records(self, limit: int = 10) -> list[dict[str, object]]:
        """Return compact recent CRM records for summary tables."""
        await self.initialize()
        safe_limit = max(1, min(100, int(limit)))
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            cursor = await conn.execute(
                """
                SELECT
                    l.id AS lead_id,
                    l.domain AS domain,
                    l.company_name AS company_name,
                    rs.id AS research_snapshot_id,
                    e.id AS email_id,
                    rs.summary AS summary,
                    e.subject AS subject,
                    ev.overall_score AS overall_score,
                    ev.created_at AS created_at
                FROM evaluations ev
                JOIN emails e ON e.id = ev.email_id
                JOIN research_snapshots rs ON rs.id = e.research_snapshot_id
                JOIN leads l ON l.id = e.lead_id
                ORDER BY ev.created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()

        records: list[dict[str, object]] = []
        for row in rows:
            records.append(
                {
                    "lead_id": int(row[0]),
                    "domain": str(row[1]),
                    "company_name": str(row[2]),
                    "research_snapshot_id": int(row[3]),
                    "email_id": int(row[4]),
                    "summary": str(row[5]),
                    "subject": str(row[6]),
                    "overall_score": float(row[7]),
                    "created_at": str(row[8]),
                }
            )
        return records

    async def full_crm_records(self, limit: int = 500) -> list[dict[str, object]]:  # noqa: C901
        """Return full-fidelity CRM records for detailed inspection tables."""
        await self.initialize()
        safe_limit = max(1, min(5000, int(limit)))
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            cursor = await conn.execute(
                """
                SELECT
                    l.id AS lead_id,
                    l.domain AS domain,
                    l.company_name AS company_name,
                    rs.id AS research_snapshot_id,
                    e.id AS email_id,
                    rs.summary AS summary,
                    rs.pain_points_json AS pain_points_json,
                    rs.value_props_json AS value_props_json,
                    rs.sources_json AS sources_json,
                    e.subject AS subject,
                    e.body AS body,
                    e.call_to_action AS call_to_action,
                    e.reflection_rounds AS reflection_rounds,
                    e.final_critique_score AS final_critique_score,
                    ev.relevance AS relevance,
                    ev.personalization AS personalization,
                    ev.tone AS tone,
                    ev.clarity AS clarity,
                    ev.rationale AS rationale,
                    ev.overall_score AS overall_score,
                    ev.created_at AS created_at
                FROM evaluations ev
                JOIN emails e ON e.id = ev.email_id
                JOIN research_snapshots rs ON rs.id = e.research_snapshot_id
                JOIN leads l ON l.id = e.lead_id
                ORDER BY ev.created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()

        records: list[dict[str, object]] = []
        for row in rows:
            try:
                pain_points = json.loads(str(row[6]))
            except Exception:  # noqa: BLE001 - tolerate legacy or malformed payloads
                pain_points = [str(row[6])]
            try:
                value_props = json.loads(str(row[7]))
            except Exception:  # noqa: BLE001 - tolerate legacy or malformed payloads
                value_props = [str(row[7])]
            try:
                sources = json.loads(str(row[8]))
            except Exception:  # noqa: BLE001 - tolerate legacy or malformed payloads
                sources = [str(row[8])]

            records.append(
                {
                    "lead_id": int(row[0]),
                    "domain": str(row[1]),
                    "company_name": str(row[2]),
                    "research_snapshot_id": int(row[3]),
                    "email_id": int(row[4]),
                    "summary": str(row[5]),
                    "pain_points": pain_points if isinstance(pain_points, list) else [str(pain_points)],
                    "value_props": value_props if isinstance(value_props, list) else [str(value_props)],
                    "sources": sources if isinstance(sources, list) else [str(sources)],
                    "subject": str(row[9]),
                    "body": str(row[10]),
                    "call_to_action": str(row[11]),
                    "reflection_rounds": int(row[12]),
                    "final_critique_score": int(row[13]),
                    "relevance": int(row[14]),
                    "personalization": int(row[15]),
                    "tone": int(row[16]),
                    "clarity": int(row[17]),
                    "rationale": str(row[18]),
                    "overall_score": float(row[19]),
                    "created_at": str(row[20]),
                }
            )
        return records

    async def eval_regression_status(self, threshold_drop: float = 0.5) -> dict[str, object]:
        """Classify model quality as stable or regressing.

        Strategy:
        - recent window: last 7 days
        - baseline window: prior 7 days
        - regression if both windows have enough data and delta <= -threshold
        """
        await self.initialize()
        now = datetime.now(timezone.utc)
        recent_since = (now - timedelta(days=7)).isoformat()
        baseline_since = (now - timedelta(days=14)).isoformat()

        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_schema(conn)
            recent_cursor = await conn.execute(
                """
                SELECT COUNT(*), AVG(overall_score)
                FROM evaluations
                WHERE created_at >= ?
                """,
                (recent_since,),
            )
            recent_row = await recent_cursor.fetchone()

            baseline_cursor = await conn.execute(
                """
                SELECT COUNT(*), AVG(overall_score)
                FROM evaluations
                WHERE created_at >= ? AND created_at < ?
                """,
                (baseline_since, recent_since),
            )
            baseline_row = await baseline_cursor.fetchone()

        recent_count = int(recent_row[0]) if recent_row and recent_row[0] is not None else 0
        baseline_count = int(baseline_row[0]) if baseline_row and baseline_row[0] is not None else 0
        recent_avg = float(recent_row[1]) if recent_row and recent_row[1] is not None else 0.0
        baseline_avg = float(baseline_row[1]) if baseline_row and baseline_row[1] is not None else 0.0

        delta = recent_avg - baseline_avg
        enough_data = recent_count >= 3 and baseline_count >= 3
        is_regressing = enough_data and delta <= -abs(float(threshold_drop))
        status = "regressing" if is_regressing else "stable"

        return {
            "status": status,
            "baseline_avg_overall_score": round(baseline_avg, 2),
            "recent_avg_overall_score": round(recent_avg, 2),
            "delta": round(delta, 2),
            "baseline_count": baseline_count,
            "recent_count": recent_count,
            "threshold_drop": abs(float(threshold_drop)),
        }
