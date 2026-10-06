"""Gmail (SMTP) による完了通知。本文には件数と論文タイトルまでを載せ、要旨は載せない。"""

from __future__ import annotations

import logging
import shutil
import smtplib
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import httpx

from .config import Config
from .pipeline import RunReport

log = logging.getLogger(__name__)


def build_message(report: RunReport, include_titles: bool, run_id: str) -> tuple[str, str]:
    final_failed = [f for f in report.failed if f["final"]]
    retrying = [f for f in report.failed if not f["final"]]
    subject = f"[paper-summarizer] 完了 {len(report.done)} 件"
    if report.failed:
        subject += f" / 失敗 {len(report.failed)} 件"
    if report.aborted:
        subject += " / 中断"

    lines = [f"ホスト: {socket.gethostname()}", f"実行 ID: {run_id}", ""]
    if report.aborted:
        lines += ["■ 実行が中断されました", f"  {report.aborted}", ""]
    if report.done:
        lines.append(f"■ 要約完了 ({len(report.done)} 件)")
        for d in report.done:
            lines.append(f"  - [{d['project']}] {d['title'] if include_titles else '#' + str(d['id'])}"
                         + (f" ({d['minutes']} 分)" if "minutes" in d else "")
                         + (f" ※{d['note']}" if "note" in d else ""))
        lines.append("")
    if final_failed:
        lines.append(f"■ 失敗 — failed/ に移動 ({len(final_failed)} 件)")
        for f in final_failed:
            lines.append(f"  - [{f['project']}] {f['name'] if include_titles else ''} : {f['error'].split(':')[0]}")
        lines.append("")
    if retrying:
        lines.append(f"■ 失敗 — 次回再試行 ({len(retrying)} 件)")
        for f in retrying:
            lines.append(f"  - [{f['project']}] {f['name'] if include_titles else ''} : {f['error'].split(':')[0]}")
        lines.append("")
    if report.duplicates:
        lines.append(f"■ 重複のためスキップ ({len(report.duplicates)} 件)")
        for d in report.duplicates:
            lines.append(f"  - [{d['project']}] {d['name'] if include_titles else ''}")
        lines.append("")
    lines.append("要約本文は同期フォルダの library/ を参照してください。詳細はログを参照。")
    return subject, "\n".join(lines)


def send(cfg: Config, subject: str, body: str) -> bool:
    n = cfg.notify
    sender, to, password = (cfg.secrets.get(k) for k in ("NOTIFY_FROM", "NOTIFY_TO", "SMTP_PASSWORD"))
    if not n.get("enabled", True):
        return False
    if not (sender and to and password):
        log.warning("メール設定 (.env の NOTIFY_FROM/NOTIFY_TO/SMTP_PASSWORD) が未設定のため通知をスキップ")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)
    try:
        with smtplib.SMTP(n.get("smtp_host", "smtp.gmail.com"), int(n.get("smtp_port", 587)), timeout=30) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        log.error("メール送信に失敗: %r", e)
        return False
    log.info("通知メール送信: %s", subject)
    return True


def notify_report(cfg: Config, report: RunReport, run_id: str, db=None) -> None:
    """処理があった回は都度通知する。1 日 1 回 (daily_report_hour 時以降の最初の実行) は稼働報告を送る。
    同じ回に両方がある場合は 1 通にまとめる。"""
    daily = db is not None and daily_report_due(cfg)
    if not report.has_news and not daily:
        return
    if report.has_news:
        subject, body = build_message(report, bool(cfg.notify.get("include_titles", True)), run_id)
    else:
        subject = "[paper-summarizer] 稼働報告 (処理対象なし)"
        body = f"ホスト: {socket.gethostname()}\n実行 ID: {run_id}\n\ninbox に未処理の論文はありませんでした。"
    if daily:
        body += "\n\n" + build_status(cfg, db)
    if send(cfg, subject, body) and daily:
        mark_daily_report_sent(cfg)


# ---- 稼働報告 -------------------------------------------------------------------
def _daily_state(cfg: Config):
    return cfg.paths.db.parent / "daily_report.date"


def daily_report_due(cfg: Config, now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.hour < int(cfg.notify.get("daily_report_hour", 6)):
        return False
    try:
        return _daily_state(cfg).read_text().strip() != now.strftime("%Y-%m-%d")
    except OSError:
        return True


def mark_daily_report_sent(cfg: Config, now: datetime | None = None) -> None:
    _daily_state(cfg).write_text((now or datetime.now()).strftime("%Y-%m-%d"))


def build_status(cfg: Config, db) -> str:
    lines = ["■ 稼働状況 (1 日 1 回)", ""]

    lines.append("プロジェクト別の件数:")
    counts: dict[str, dict[str, int]] = {}
    for r in db.status_counts():
        counts.setdefault(r["project"], {})[r["status"]] = r["n"]
    if counts:
        for proj, c in sorted(counts.items()):
            lines.append(f"  - {proj}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    else:
        lines.append("  (登録なし)")

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).astimezone().isoformat(timespec="seconds")
    runs = db.conn.execute("SELECT status, n_done, n_failed FROM runs WHERE kind='run' AND started_at >= ?",
                           (since,)).fetchall()
    errors = sum(1 for r in runs if r["status"] not in ("ok", None))
    lines += ["", f"直近 24 時間の実行: {len(runs)} 回 (完了 {sum(r['n_done'] or 0 for r in runs)} 本, "
                  f"失敗 {sum(r['n_failed'] or 0 for r in runs)} 本, 異常終了 {errors} 回)"]

    models = [cfg.ollama["model"], cfg.ollama.get("vision_model", cfg.ollama["model"])]
    try:
        r = httpx.get(cfg.ollama["url"].rstrip("/") + "/api/tags", timeout=10)
        names = {m["name"] for m in r.json().get("models", [])}
        missing = [m for m in set(models) if m not in names]
        lines.append("Ollama: 応答あり" + (f" (モデルが見つかりません: {missing})" if missing else f" / モデル {models[0]} あり"))
    except (httpx.HTTPError, ValueError) as e:
        lines.append(f"Ollama: 応答なし ({type(e).__name__})")

    du = shutil.disk_usage(cfg.paths.root)
    lines.append(f"ディスク ({cfg.paths.root}): 空き {du.free / 1e9:.0f} GB / {du.total / 1e9:.0f} GB")

    nxt = _next_timer()
    if nxt:
        lines.append(f"次回の実行: {nxt}")
    return "\n".join(lines)


def _next_timer() -> str | None:
    try:
        out = subprocess.run(["systemctl", "show", "paper-summarizer.timer", "-p", "NextElapseUSecRealtime",
                              "--value"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None
