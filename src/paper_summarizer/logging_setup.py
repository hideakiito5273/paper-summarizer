"""ログ設定: 日次ローテーションのファイルログ + 標準エラー (systemd 経由で journald)。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def setup_logging(log_dir: Path, run_id: str, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(FORMAT)
    run_filter = _RunIdFilter(run_id)

    file_handler = TimedRotatingFileHandler(
        log_dir / "paper-summarizer.log", when="midnight", backupCount=180, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(run_filter)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(fmt)
    stream.addFilter(run_filter)
    root.addHandler(stream)

    for noisy in ("httpx", "httpcore", "docling", "urllib3", "PIL", "rapidocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
