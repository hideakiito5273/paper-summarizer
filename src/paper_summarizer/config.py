"""設定ファイル (config.toml) と秘密情報 (.env) の読み込み。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_DIR / "prompts"


@dataclass(frozen=True)
class Paths:
    root: Path
    db: Path
    log_dir: Path
    work_dir: Path

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def library(self) -> Path:
        return self.root / "library"

    @property
    def reviews(self) -> Path:
        return self.root / "reviews"

    @property
    def failed(self) -> Path:
        return self.root / "failed"


@dataclass(frozen=True)
class Config:
    paths: Paths
    scan: dict
    ollama: dict
    extract: dict
    summarize: dict
    notify: dict
    secrets: dict


def load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    # 環境変数が設定されていればそちらを優先 (systemd の EnvironmentFile 経由など)
    for key in ("NOTIFY_FROM", "NOTIFY_TO", "SMTP_PASSWORD"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def load_config(path: Path | None = None) -> Config:
    path = path or Path(os.environ.get("PAPER_SUMMARIZER_CONFIG", PROJECT_DIR / "config.toml"))
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    p = raw["paths"]
    paths = Paths(
        root=Path(p["root"]).expanduser(),
        db=Path(p["db"]).expanduser(),
        log_dir=Path(p["log_dir"]).expanduser(),
        work_dir=Path(p["work_dir"]).expanduser(),
    )
    return Config(
        paths=paths,
        scan=raw.get("scan", {}),
        ollama=raw["ollama"],
        extract=raw.get("extract", {}),
        summarize=raw.get("summarize", {}),
        notify=raw.get("notify", {}),
        secrets=load_env(PROJECT_DIR / ".env"),
    )


def ensure_dirs(cfg: Config) -> None:
    for d in (
        cfg.paths.inbox / "_unsorted",
        cfg.paths.library,
        cfg.paths.reviews,
        cfg.paths.failed,
        cfg.paths.db.parent,
        cfg.paths.work_dir,
        cfg.paths.log_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
