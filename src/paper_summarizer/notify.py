"""Gmail (SMTP) による完了通知。本文には件数と論文タイトルまでを載せ、要旨は載せない。"""

from __future__ import annotations

import logging
import smtplib
import socket
from email.message import EmailMessage

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


def notify_report(cfg: Config, report: RunReport, run_id: str) -> None:
    if not report.has_news:
        return
    subject, body = build_message(report, bool(cfg.notify.get("include_titles", True)), run_id)
    send(cfg, subject, body)
