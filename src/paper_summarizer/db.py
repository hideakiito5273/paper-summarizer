"""SQLite による処理状態管理。

papers.status の遷移:
  pending → processing → done
                       ↘ pending (失敗、再試行回数内) → ... → failed
  pending → duplicate (処理済みと同一内容)
  pending → missing   (処理前に inbox から消えた)
  done    → superseded (同じプロジェクト・同じファイル名の差し替え版が処理された)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id            INTEGER PRIMARY KEY,
    sha256        TEXT NOT NULL,
    project       TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    inbox_path    TEXT,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    replaces      INTEGER REFERENCES papers(id),
    title         TEXT,
    authors       TEXT,
    year          INTEGER,
    venue         TEXT,
    doi           TEXT,
    output_dir    TEXT,
    model         TEXT,
    prompt_version TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    duration_s    REAL
);
CREATE INDEX IF NOT EXISTS idx_papers_sha ON papers(sha256);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_proj_name ON papers(project, source_name);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT,
    n_done      INTEGER DEFAULT 0,
    n_failed    INTEGER DEFAULT 0,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT,
    paper_id      INTEGER,
    stage         TEXT NOT NULL,
    model         TEXT NOT NULL,
    prompt_tokens INTEGER,
    eval_tokens   INTEGER,
    duration_s    REAL,
    ok            INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        # LLM 呼び出しを並列実行するため、複数スレッドから使う (書き込みは lock で直列化)
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=30, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(llm_calls)")}
        for col, typ in (("load_s", "REAL"), ("prefill_s", "REAL"), ("decode_s", "REAL"), ("think", "TEXT")):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE llm_calls ADD COLUMN {col} {typ}")

    # ---- papers -------------------------------------------------------
    def get(self, paper_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()

    def find_by_sha(self, sha: str, statuses: tuple[str, ...]) -> sqlite3.Row | None:
        q = f"SELECT * FROM papers WHERE sha256=? AND status IN ({','.join('?' * len(statuses))}) ORDER BY id DESC"
        return self.conn.execute(q, (sha, *statuses)).fetchone()

    def find_by_inbox_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM papers WHERE inbox_path=? AND status IN ('pending','processing') ORDER BY id DESC",
            (path,),
        ).fetchone()

    def find_done_by_name(self, project: str, source_name: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM papers WHERE project=? AND source_name=? AND status='done' ORDER BY id DESC",
            (project, source_name),
        ).fetchone()

    def add(self, *, sha256: str, project: str, source_name: str, inbox_path: str,
            status: str = "pending", replaces: int | None = None, last_error: str | None = None) -> int:
        t = now()
        cur = self.conn.execute(
            "INSERT INTO papers (sha256, project, source_name, inbox_path, status, replaces, last_error,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (sha256, project, source_name, inbox_path, status, replaces, last_error, t, t),
        )
        return cur.lastrowid

    def update(self, paper_id: int, **fields) -> None:
        if "authors" in fields and not isinstance(fields["authors"], (str, type(None))):
            fields["authors"] = json.dumps(fields["authors"], ensure_ascii=False)
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE papers SET {cols} WHERE id=?", (*fields.values(), paper_id))

    def pending(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM papers WHERE status='pending' ORDER BY id").fetchall()

    def reset_stale_processing(self) -> int:
        """前回の実行が途中で落ちた場合に processing のまま残った行を pending に戻す。"""
        cur = self.conn.execute(
            "UPDATE papers SET status='pending', updated_at=? WHERE status='processing'", (now(),)
        )
        return cur.rowcount

    def done_in_project(self, project: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM papers WHERE project=? AND status='done' ORDER BY year, title", (project,)
        ).fetchall()

    def status_counts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT project, status, COUNT(*) AS n FROM papers GROUP BY project, status ORDER BY project, status"
        ).fetchall()

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM papers ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()

    # ---- runs / llm calls --------------------------------------------
    def start_run(self, run_id: str, kind: str) -> None:
        self.conn.execute("INSERT INTO runs (id, kind, started_at) VALUES (?,?,?)", (run_id, kind, now()))

    def finish_run(self, run_id: str, status: str, n_done: int = 0, n_failed: int = 0,
                   error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, n_done=?, n_failed=?, error=? WHERE id=?",
            (now(), status, n_done, n_failed, error, run_id),
        )

    def log_llm_call(self, *, run_id: str | None, paper_id: int | None, stage: str, model: str,
                     prompt_tokens: int | None, eval_tokens: int | None, duration_s: float, ok: bool,
                     load_s: float | None = None, prefill_s: float | None = None, decode_s: float | None = None,
                     think: str | None = None) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO llm_calls (run_id, paper_id, stage, model, prompt_tokens, eval_tokens, duration_s, ok,"
                " created_at, load_s, prefill_s, decode_s, think) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, paper_id, stage, model, prompt_tokens, eval_tokens, duration_s, int(ok), now(),
                 load_s, prefill_s, decode_s, think),
            )
