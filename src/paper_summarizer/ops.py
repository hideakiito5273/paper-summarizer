"""CLI と Web の両方から使う操作 (再試行・再処理・実行の起動・アップロードの保存)。"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_DIR, Config
from .db import DB

log = logging.getLogger(__name__)

PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class OpError(Exception):
    """利用者に表示してよい操作エラー。"""


# ---- 多重起動防止 -----------------------------------------------------------
def run_lock(cfg: Config):
    """取得できればファイルオブジェクト (保持している間ロック)、実行中なら None。"""
    f = open(cfg.paths.db.parent / "run.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def is_running(cfg: Config) -> bool:
    lock = run_lock(cfg)
    if lock is None:
        return True
    lock.close()
    return False


def start_run_detached(cfg: Config) -> bool:
    """paper-summarizer run をセッションから切り離して起動する。実行中なら False。"""
    if is_running(cfg):
        return False
    exe = Path(sys.executable).with_name("paper-summarizer")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "paper_summarizer.cli"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    log_path = cfg.paths.log_dir / f"manual-run-{time.strftime('%Y%m%d-%H%M%S')}.out"
    with open(log_path, "w") as out:
        subprocess.Popen(cmd + ["run"], cwd=PROJECT_DIR, env=env, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log.info("手動実行を起動しました (%s)", log_path)
    return True


# ---- 再試行・再処理 -----------------------------------------------------------
def retry(cfg: Config, db: DB, paper_id: int) -> Path:
    """failed / pending の論文を再キューする。"""
    row = db.get(paper_id)
    if row is None or row["status"] not in ("failed", "pending"):
        raise OpError(f"#{paper_id} は再試行できる状態ではありません")
    src = Path(row["inbox_path"])
    if row["status"] == "failed":
        dest_dir = cfg.paths.inbox / row["project"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / row["source_name"]
        if dest.exists():
            raise OpError(f"inbox に同名ファイルがあります: {dest.name}")
        shutil.move(str(src), dest)
        src.with_name(src.name + ".error.txt").unlink(missing_ok=True)
        src = dest
    db.update(paper_id, status="pending", attempts=0, inbox_path=str(src), last_error=None)
    log.info("#%d を再キューしました: %s", paper_id, src)
    return src


def reprocess(cfg: Config, db: DB, paper_id: int, user: str | None = None) -> int:
    """処理済みの論文を再要約する。旧版は処理成功時に _history/ に退避される。新しい論文 ID を返す。"""
    row = db.get(paper_id)
    if row is None or row["status"] != "done":
        raise OpError(f"#{paper_id} は処理済み (done) ではありません")
    src = Path(row["output_dir"]) / "paper.pdf"
    if not src.exists():
        raise OpError(f"PDF が見つかりません: {src}")
    dest = cfg.paths.inbox / row["project"] / row["source_name"]
    if dest.exists():
        raise OpError(f"inbox に同名ファイルがあります: {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    pid = db.add(sha256=row["sha256"], project=row["project"], source_name=row["source_name"],
                 inbox_path=str(dest), replaces=row["id"])
    if user:
        db.update(pid, submitted_by=user)
    log.info("#%d を再処理対象に登録しました → #%d (%s)", row["id"], pid, dest)
    return pid


# ---- アップロード -------------------------------------------------------------
@dataclass
class UploadResult:
    name: str
    project: str
    ok: bool
    message: str


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name  # パス成分を除く
    name = re.sub(r"[^\w.\-() ]+", "_", name).strip(" .")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:150] or "upload.pdf"


def save_upload(cfg: Config, db: DB, project: str, filename: str, data: bytes, user: str) -> UploadResult:
    """inbox/<project>/ に原子的に保存する (書き込み途中のファイルを走査で拾わないよう、一時名 → rename)。"""
    if not PROJECT_RE.match(project):
        raise OpError("プロジェクト名は英数字・ハイフン・アンダースコア (64 文字以内) で指定してください")
    name = safe_filename(filename)
    if len(data) > MAX_UPLOAD_BYTES:
        return UploadResult(name, project, False, f"サイズが上限 ({MAX_UPLOAD_BYTES // 1024 // 1024} MB) を超えています")
    if not data.startswith(b"%PDF"):
        return UploadResult(name, project, False, "PDF ファイルではありません")

    sha = hashlib.sha256(data).hexdigest()
    done = db.find_by_sha(sha, ("done",))
    if done and done["project"] == project:
        return UploadResult(name, project, False, f"同じ内容の論文が処理済みです (#{done['id']} {done['title'] or ''})")
    queued = db.find_by_sha(sha, ("pending", "processing"))
    if queued:
        return UploadResult(name, project, False, f"同じ内容の論文が処理待ちです (#{queued['id']})")

    dest_dir = cfg.paths.inbox / project
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        stem, k = dest.stem, 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{k}.pdf"
            k += 1
    tmp = dest_dir / f".{dest.name}.uploading"
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    # 書き込み完了済みなので、同期途中を避けるための待ち時間 (min_age_seconds) の対象外にする
    past = time.time() - 3600
    os.utime(dest, (past, past))
    db.add_upload(sha256=sha, path=str(dest), user=user)

    note = "受け付けました"
    if done:
        note += f" (他プロジェクト {done['project']} の処理済み要約を複製します)"
    elif db.find_done_by_name(project, dest.name):
        note += " (同名の論文の差し替えとして再要約します)"
    log.info("アップロード: %s/%s by %s (%d bytes)", project, dest.name, user, len(data))
    return UploadResult(dest.name, project, True, note)
