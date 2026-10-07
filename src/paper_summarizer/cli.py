"""コマンドライン: paper-summarizer {run,status,review,retry,readme,try,test-mail}"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .config import Config, ensure_dirs, load_config
from .db import DB
from .llm import OllamaClient
from .logging_setup import setup_logging

log = logging.getLogger("paper_summarizer")


def _run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
def cmd_run(cfg: Config, db: DB, run_id: str, args) -> int:
    from .notify import notify_report
    from .pipeline import RunReport, process_pending, scan

    from .ops import run_lock

    lock = run_lock(cfg)
    if lock is None:
        log.info("別の実行が進行中のため終了します")
        return 0
    db.start_run(run_id, "run")
    report = RunReport()
    try:
        n = db.reset_stale_processing()
        if n:
            log.warning("前回中断された処理 %d 件を再キューしました", n)
        scan(cfg, db, report)
        llm = OllamaClient(cfg.ollama, db=db, run_id=run_id)
        process_pending(cfg, db, llm, report)
    except Exception as e:
        log.exception("実行全体が異常終了しました")
        report.aborted = f"{type(e).__name__}: {e}"
        db.finish_run(run_id, "error", len(report.done), len(report.failed), report.aborted)
        if not args.no_notify:
            notify_report(cfg, report, run_id, db)
        return 1
    status = "aborted" if report.aborted else "ok"
    db.finish_run(run_id, status, len(report.done), len(report.failed), report.aborted)
    log.info("実行終了: 完了 %d / 失敗 %d / 重複 %d", len(report.done), len(report.failed), len(report.duplicates))
    if not args.no_notify:
        notify_report(cfg, report, run_id, db)
    return 0 if not report.aborted else 1


def cmd_status(cfg: Config, db: DB, run_id: str, args) -> int:
    print("== プロジェクト別件数")
    for r in db.status_counts():
        print(f"  {r['project']:<24} {r['status']:<11} {r['n']}")
    print("\n== 最近の更新")
    for r in db.recent(args.limit):
        title = (r["title"] or r["source_name"])[:60]
        err = f"  ! {r['last_error'][:80]}" if r["last_error"] and r["status"] != "done" else ""
        print(f"  #{r['id']:<4} {r['status']:<11} {r['updated_at'][:16]}  [{r['project']}] {title}{err}")
    return 0


def cmd_review(cfg: Config, db: DB, run_id: str, args) -> int:
    from .review import run_review

    llm_cfg = dict(cfg.ollama, num_ctx=int(cfg.summarize.get("review_num_ctx", 131072)))
    db.start_run(run_id, "review")
    try:
        path = run_review(cfg, db, OllamaClient(llm_cfg, db=db, run_id=run_id), args.project)
    except Exception as e:
        db.finish_run(run_id, "error", error=f"{type(e).__name__}: {e}")
        raise
    db.finish_run(run_id, "ok")
    print(path)
    return 0


def cmd_retry(cfg: Config, db: DB, run_id: str, args) -> int:
    from .ops import OpError, retry

    try:
        retry(cfg, db, args.id)
    except OpError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def cmd_reprocess(cfg: Config, db: DB, run_id: str, args) -> int:
    """処理済みの論文を再要約する (プロンプト・構成の変更後や検証失敗時)。旧版は _history/ に退避される。"""
    from .ops import OpError, reprocess

    try:
        reprocess(cfg, db, args.id)
    except OpError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def cmd_readme(cfg: Config, db: DB, run_id: str, args) -> int:
    from .output import update_project_readme

    print(update_project_readme(db, cfg.paths.reviews, cfg.paths.library, args.project))
    return 0


def cmd_try(cfg: Config, db: DB, run_id: str, args) -> int:
    """DB・フォルダ移動なしで 1 本だけ要約する (モデル比較・プロンプト調整用)。"""
    import time

    from .extract import Extractor
    from .output import write_reading_outputs
    from .summarize import summarize

    ollama_cfg = dict(cfg.ollama)
    if args.model:
        ollama_cfg["model"] = ollama_cfg["vision_model"] = args.model
    llm = OllamaClient(ollama_cfg, db=db, run_id=run_id)
    llm.check([ollama_cfg["model"], ollama_cfg["vision_model"]])
    out = Path(args.out)
    t0 = time.monotonic()
    ext = Extractor(cfg.extract).extract(Path(args.pdf), out / "work", llm, ollama_cfg["vision_model"])
    t1 = time.monotonic()
    result = summarize(ext, Path(args.pdf).stem, llm, cfg.summarize)
    t2 = time.monotonic()
    (out / "summary.md").write_text(result.markdown, encoding="utf-8")
    write_reading_outputs(out, Path(args.pdf).stem, ext, result)
    (out / "verification.json").write_text(
        json.dumps({"converged": result.converged, "rounds": result.rounds}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    stats = {"model": ollama_cfg["model"], "parallel": llm.parallel,
             "think_levels": ollama_cfg.get("think_levels", {}), "extract_min": round((t1 - t0) / 60, 1),
             "summarize_min": round((t2 - t1) / 60, 1), "chunks": result.n_chunks,
             "figures": len(ext.figures), "verify_rounds": len(result.rounds), "converged": result.converged,
             "issues_per_round": [len(r["issues"]) for r in result.rounds],
             "stages": stage_stats(db, run_id)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def stage_stats(db: DB, run_id: str) -> dict:
    """段階ごとの呼び出し回数・トークン数・時間の内訳 (性能比較用)。"""
    rows = db.conn.execute(
        "SELECT CASE WHEN instr(stage, ':') > 0 THEN substr(stage, 1, instr(stage, ':') - 1) ELSE stage END AS s,"
        " COUNT(*) AS calls, SUM(prompt_tokens) AS prompt_tok, SUM(eval_tokens) AS eval_tok,"
        " ROUND(SUM(duration_s) / 60, 1) AS wall_min, ROUND(SUM(load_s), 1) AS load_s,"
        " ROUND(SUM(prefill_s), 1) AS prefill_s, ROUND(SUM(decode_s), 1) AS decode_s,"
        " ROUND(SUM(eval_tokens) / NULLIF(SUM(decode_s), 0), 1) AS decode_tok_s"
        " FROM llm_calls WHERE run_id=? AND ok=1 GROUP BY s ORDER BY SUM(duration_s) DESC", (run_id,)).fetchall()
    return {r["s"]: {k: r[k] for k in r.keys() if k != "s"} for r in rows}


def cmd_web(cfg: Config, db: DB, run_id: str, args) -> int:
    import uvicorn

    from .web.app import create_app

    w = cfg.web
    ssl = {}
    if w.get("https", True):
        cert, key = Path(w["cert"]), Path(w["key"])
        if not (cert.exists() and key.exists()):
            print("証明書がありません。先に paper-summarizer web-cert を実行してください", file=sys.stderr)
            return 1
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    log.info("Web UI を起動: %s://%s:%s", "https" if ssl else "http", w.get("host", "0.0.0.0"), w.get("port", 8443))
    uvicorn.run(create_app(cfg), host=w.get("host", "0.0.0.0"), port=int(w.get("port", 8443)),
                log_config=None, access_log=False, **ssl)
    return 0


def cmd_web_cert(cfg: Config, db: DB, run_id: str, args) -> int:
    """LAN / VPN 内向けの自己署名証明書を作る (ブラウザで初回に警告が出る)。"""
    import socket
    import subprocess

    cert, key = Path(cfg.web["cert"]), Path(cfg.web["key"])
    cert.parent.mkdir(parents=True, exist_ok=True)
    ips = args.ip or subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    ips = [ip for ip in ips if not ip.startswith("172.17.")]  # docker0 は除く
    san = ",".join([f"DNS:{socket.gethostname()}", "DNS:localhost", "IP:127.0.0.1"] + [f"IP:{ip}" for ip in ips])
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-days", "825", "-keyout", str(key), "-out", str(cert),
                    "-subj", f"/CN={socket.gethostname()}", "-addext", f"subjectAltName={san}"], check=True,
                   capture_output=True)
    key.chmod(0o600)
    print(f"作成しました: {cert}\n  対象: {san}")
    return 0


def cmd_user(cfg: Config, db: DB, run_id: str, args) -> int:
    import getpass

    from .web import auth

    state = cfg.paths.db.parent
    if args.action == "list":
        for name, u in sorted(auth.load_users(state).items()):
            print(f"{name}\t{u.get('name', '')}")
        return 0
    if not args.username:
        print("ユーザー名を指定してください", file=sys.stderr)
        return 1
    if args.action == "remove":
        ok = auth.remove_user(state, args.username)
        print("削除しました" if ok else "見つかりません")
        return 0 if ok else 1
    pw = getpass.getpass("パスワード: ")
    if pw != getpass.getpass("パスワード (確認): "):
        print("パスワードが一致しません", file=sys.stderr)
        return 1
    try:
        auth.set_password(state, args.username, pw, args.name)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{'更新' if args.action == 'passwd' else '追加'}しました: {args.username}")
    return 0


def cmd_test_mail(cfg: Config, db: DB, run_id: str, args) -> int:
    from .notify import send

    ok = send(cfg, "[paper-summarizer] テスト送信", "paper-summarizer からのテストメールです。")
    print("送信しました" if ok else "送信できませんでした (ログを確認)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="paper-summarizer")
    p.add_argument("--config", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="inbox を走査して未処理の論文を要約 (定期実行用)")
    s.add_argument("--no-notify", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="処理状況の表示")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("review", help="プロジェクト単位の文献レビューを作成")
    s.add_argument("project")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("retry", help="failed の論文を再キュー")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_retry)

    s = sub.add_parser("reprocess", help="処理済みの論文を再要約 (旧版は _history/ に退避)")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_reprocess)

    s = sub.add_parser("readme", help="プロジェクトの文献一覧 README を再生成")
    s.add_argument("project")
    s.set_defaults(func=cmd_readme)

    s = sub.add_parser("try", help="1 本だけ試験的に要約 (DB・ファイル移動なし)")
    s.add_argument("pdf")
    s.add_argument("--out", required=True)
    s.add_argument("--model")
    s.set_defaults(func=cmd_try)

    s = sub.add_parser("web", help="Web UI を起動 (LAN / 公式 VPN 内向け)")
    s.set_defaults(func=cmd_web)

    s = sub.add_parser("web-cert", help="Web UI 用の自己署名証明書を作成")
    s.add_argument("--ip", action="append", help="証明書に含める IP (省略時は hostname -I)")
    s.set_defaults(func=cmd_web_cert)

    s = sub.add_parser("user", help="Web UI のユーザー管理")
    s.add_argument("action", choices=["add", "passwd", "remove", "list"])
    s.add_argument("username", nargs="?")
    s.add_argument("--name", help="表示名")
    s.set_defaults(func=cmd_user)

    s = sub.add_parser("test-mail", help="通知メールのテスト送信")
    s.set_defaults(func=cmd_test_mail)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    run_id = _run_id()
    setup_logging(cfg.paths.log_dir, run_id, args.verbose)
    db = DB(cfg.paths.db)
    log.info("コマンド開始: %s", " ".join(sys.argv[1:]) if argv is None else " ".join(argv))
    return args.func(cfg, db, run_id, args)


if __name__ == "__main__":
    sys.exit(main())
