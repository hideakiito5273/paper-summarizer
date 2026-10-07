"""FastAPI アプリ。要約処理そのものは systemd timer / 手動実行 (paper-summarizer run) が担い、
ここでは inbox への投稿、状態 DB の参照、成果物の閲覧、再試行・再処理の受付だけを行う。"""

from __future__ import annotations

import html
import json
import logging
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path

import httpx
import markdown as md
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .. import ops
from ..config import Config, load_config
from ..db import DB
from ..notify import _next_timer
from ..output import first_point
from . import auth

log = logging.getLogger(__name__)
access = logging.getLogger("paper_summarizer.web.access")
HERE = Path(__file__).parent

STAGE_LABELS = {
    "figure": "図の読み取り", "formula": "数式の読み取り", "metadata": "書誌情報の抽出",
    "notes": "精読", "merge": "統合", "shorten": "短縮", "verify": "照合", "revise": "修正",
}
STATUS_LABELS = {
    "pending": "処理待ち", "processing": "処理中", "done": "完了", "failed": "失敗",
    "duplicate": "重複", "missing": "消失", "superseded": "旧版",
}
CITE_RE = re.compile(r"\[(§[^\]]+)\]")
CITE_ID_RE = re.compile(r"§[\w.]+-p\d+")
REVIEW_RE = re.compile(r"^review_[\w-]+\.md$")


def client_ip(request: Request) -> str:
    """接続元 IP。Web UI はプロキシを介さず直接受けるため、X-Forwarded-For は信用しない。"""
    return request.client.host if request.client else "?"


# ---- 表示用ヘルパー -------------------------------------------------------------
def fmt_time(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%m/%d %H:%M")
    except ValueError:
        return iso[:16]


def fmt_minutes(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds / 60:.0f} 分"


def elapsed_since(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        sec = (datetime.now().astimezone() - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return ""
    return f"{sec / 60:.0f} 分" if sec < 5400 else f"{sec / 3600:.1f} 時間"


def stage_label(stage: str) -> str:
    base, _, rest = stage.partition(":")
    label = STAGE_LABELS.get(base, base)
    if base in ("verify", "revise") and rest.startswith("r"):
        label += f" {rest[1:].split(':')[0]} 回目"
    elif base == "notes" and rest:
        label += f" ({rest})"
    return label


_STAGE_ORDER = ["formula", "figure", "metadata", "notes", "merge", "shorten", "verify", "revise"]


def stage_progress(stage: str) -> float:
    """最後に完了した段階から、おおよその進み具合 (0〜1) を出す。照合・修正は繰り返すため 0.9 で頭打ち。"""
    base, _, rest = stage.partition(":")
    if base not in _STAGE_ORDER:
        return 0.05
    frac = (_STAGE_ORDER.index(base) + 1) / len(_STAGE_ORDER)
    m = re.match(r"(\d+)/(\d+)", rest)
    if base == "notes" and m:  # 精読は i/n で細かく
        start = _STAGE_ORDER.index("notes") / len(_STAGE_ORDER)
        frac = start + (int(m.group(1)) / int(m.group(2))) / len(_STAGE_ORDER)
    return min(frac, 0.9)


def render_markdown(text: str) -> str:
    """要約などの Markdown を HTML にする。数式 ($…$, $$…$$) は KaTeX に任せるため退避し、
    根拠表示 [§3.2-p4] はクリックで原文を開けるリンクにする。"""
    stash: list[str] = []

    def keep(m: re.Match) -> str:
        stash.append(html.escape(m.group(0)))
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", keep, text, flags=re.S)

    def cite(m: re.Match) -> str:
        ids = CITE_ID_RE.findall(m.group(1))
        if not ids:
            return m.group(0)
        stash.append(f'<a href="#" class="cite" data-ids="{html.escape(",".join(ids))}">'
                     f'[{html.escape(m.group(1))}]</a>')
        return f"\x00{len(stash) - 1}\x00"

    text = CITE_RE.sub(cite, text)
    out = md.markdown(text, extensions=["tables", "sane_lists", "fenced_code"])
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], out)


def strip_title_block(text: str) -> str:
    """ページ上部に書誌情報を出すため、Markdown 冒頭のタイトル (#) と書誌表を除く。"""
    m = re.search(r"(?m)^## ", text)
    return text[m.start():] if m else text


def _safe_output_dir(cfg: Config, row) -> Path:
    out = Path(row["output_dir"] or "")
    if not row["output_dir"] or not out.resolve().is_relative_to(cfg.paths.library.resolve()):
        raise HTTPException(404, "成果物が見つかりません")
    return out


# ---- 状態の集計 ---------------------------------------------------------------
_health_cache: dict = {"at": 0.0, "data": None}


def health(cfg: Config) -> dict:
    if time.time() - _health_cache["at"] < 60 and _health_cache["data"]:
        return _health_cache["data"]
    data = {"ollama": False, "model": False, "disk_free_gb": None, "next_run": _next_timer()}
    try:
        r = httpx.get(cfg.ollama["url"].rstrip("/") + "/api/tags", timeout=5)
        names = {m["name"] for m in r.json().get("models", [])}
        data["ollama"] = True
        data["model"] = cfg.ollama["model"] in names
    except (httpx.HTTPError, ValueError):
        pass
    du = shutil.disk_usage(cfg.paths.root)
    data["disk_free_gb"] = round(du.free / 1e9)
    _health_cache.update(at=time.time(), data=data)
    return data


def status_data(cfg: Config, db: DB) -> dict:
    processing = []
    for row in db.by_status(("processing",), 10):
        call = db.last_llm_call(row["id"])
        started = row["started_at"]
        if call is None or (started and call["created_at"] < started):
            stage, progress = "PDF の抽出 (Docling)", 0.03
        else:
            stage, progress = f"{stage_label(call['stage'])} まで完了", stage_progress(call["stage"])
        processing.append({"row": row, "stage": stage, "progress": progress, "elapsed": elapsed_since(started)})

    counts: dict[str, dict[str, int]] = {}
    for r in db.status_counts():
        counts.setdefault(r["project"], {})[r["status"]] = r["n"]
    return {
        "running": ops.is_running(cfg),
        "processing": processing,
        "pending": db.by_status(("pending",), 50, newest_first=False),
        "done": db.by_status(("done",), 10),
        "failed": db.by_status(("failed",), 20),
        "runs": db.recent_runs(6),
        "counts": counts,
        "health": health(cfg),
    }


# ---- アプリ -------------------------------------------------------------------
def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state_dir = cfg.paths.db.parent
    db = DB(cfg.paths.db)
    app = FastAPI(title="paper-summarizer", docs_url=None, redoc_url=None, openapi_url=None)
    # アクセスログ (接続元 IP・ユーザー・パス・応答コード・時間)。セッションを読むため SessionMiddleware の内側に置く
    #  (add_middleware は後から追加したものが外側になるので、Session より先に登録する)
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = request.url.path
            user = request.session.get("user", "-") if "session" in request.scope else "-"
            quiet = path.startswith("/static/") or path == "/fragment/status"  # 自動更新・静的ファイルは詳細ログのみ
            (access.debug if quiet else access.info)(
                "%s %s %s %d user=%s %.0fms", client_ip(request), request.method, path, status, user,
                (time.monotonic() - t0) * 1000)

    app.add_middleware(SessionMiddleware, secret_key=auth.secret_key(state_dir), session_cookie="ps_session",
                       max_age=14 * 24 * 3600, same_site="lax", https_only=cfg.web.get("https", True))
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters.update(t=fmt_time, minutes=fmt_minutes, md=render_markdown, stage=stage_label)
    templates.env.globals.update(STATUS_LABELS=STATUS_LABELS)

    # -- 認証・CSRF --
    def user_of(request: Request) -> str | None:
        user = request.session.get("user")
        return user if user and user in auth.load_users(state_dir) else None

    def csrf_token(request: Request) -> str:
        if "csrf" not in request.session:
            request.session["csrf"] = secrets.token_hex(16)
        return request.session["csrf"]

    def require_user(request: Request) -> str:
        user = user_of(request)
        if user is None:
            raise HTTPException(401, "ログインが必要です")
        return user

    def require_csrf(request: Request, token: str | None) -> None:
        if not token or not secrets.compare_digest(token, request.session.get("csrf", "")):
            raise HTTPException(403, "不正なリクエストです (CSRF)")

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        user = user_of(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        users = auth.load_users(state_dir)
        return templates.TemplateResponse(request, name, {"user": user, "user_name": users[user].get("name", user),
                                                          "csrf": csrf_token(request), **ctx})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith(("/api/", "/fragment/")):
            return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)
        if exc.status_code == 401:
            return RedirectResponse("/login", status_code=303)
        return HTMLResponse(f"<p>{html.escape(str(exc.detail))}</p><p><a href='/'>戻る</a></p>",
                            status_code=exc.status_code)

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, error: str = ""):
        return templates.TemplateResponse(request, "login.html", {"error": error, "csrf": csrf_token(request)})

    @app.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...), csrf: str = Form("")):
        require_csrf(request, csrf)
        if not auth.verify(state_dir, username.strip(), password):
            log.warning("ログイン失敗: %s (%s)", username, client_ip(request))
            return RedirectResponse("/login?error=1", status_code=303)
        request.session.clear()
        request.session.update(user=username.strip(), csrf=secrets.token_hex(16))
        log.info("ログイン: %s (%s)", username, client_ip(request))
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    def logout(request: Request, csrf: str = Form("")):
        require_csrf(request, csrf)
        log.info("ログアウト: %s (%s)", request.session.get("user", "-"), client_ip(request))
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # -- ダッシュボード --
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard.html", s=status_data(cfg, db))

    @app.get("/fragment/status", response_class=HTMLResponse)
    def status_fragment(request: Request):
        require_user(request)
        return templates.TemplateResponse(request, "_status.html", {"s": status_data(cfg, db)})

    @app.post("/api/run")
    def run_now(request: Request):
        user = require_user(request)
        require_csrf(request, request.headers.get("x-csrf-token"))
        if not ops.start_run_detached(cfg):
            return {"ok": False, "error": "すでに実行中です"}
        log.info("今すぐ処理: %s", user)
        return {"ok": True}

    # -- 投稿 --
    @app.get("/upload", response_class=HTMLResponse)
    def upload_form(request: Request):
        projects = sorted(set(db.projects()) | {p.name for p in cfg.paths.inbox.iterdir()
                                                 if p.is_dir() and not p.name.startswith((".", "_"))})
        return page(request, "upload.html", projects=projects)

    @app.post("/api/upload")
    async def upload(request: Request, project: str = Form(...), files: list[UploadFile] = File(...)):
        user = require_user(request)
        require_csrf(request, request.headers.get("x-csrf-token"))
        results = []
        for f in files:
            data = await f.read(ops.MAX_UPLOAD_BYTES + 1)
            try:
                r = ops.save_upload(cfg, db, project.strip(), f.filename or "upload.pdf", data, user)
            except ops.OpError as e:
                raise HTTPException(400, str(e)) from e
            results.append(r.__dict__)
        return {"ok": True, "results": results}

    # -- ライブラリ --
    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        counts: dict[str, dict[str, int]] = {}
        for r in db.status_counts():
            counts.setdefault(r["project"], {})[r["status"]] = r["n"]
        return page(request, "library.html", counts=counts)

    @app.get("/library/{project}", response_class=HTMLResponse)
    def project_page(request: Request, project: str):
        if not ops.PROJECT_RE.match(project):
            raise HTTPException(404)
        papers = []
        for r in db.done_in_project(project):
            authors = json.loads(r["authors"]) if r["authors"] else []
            out = Path(r["output_dir"]) if r["output_dir"] else None
            papers.append({"row": r, "first_author": authors[0] if authors else "",
                           "gist": first_point(out / "summary.md") if out else ""})
        rev_dir = cfg.paths.reviews / project
        reviews = sorted((p.name for p in rev_dir.glob("review_*.md")), reverse=True) if rev_dir.exists() else []
        return page(request, "project.html", project=project, papers=papers, reviews=reviews)

    @app.get("/library/{project}/review/{name}", response_class=HTMLResponse)
    def review_page(request: Request, project: str, name: str):
        path = cfg.paths.reviews / project / name
        if not ops.PROJECT_RE.match(project) or not REVIEW_RE.match(name) or not path.exists():
            raise HTTPException(404)
        return page(request, "document.html", title=f"{project} / {name}", project=project,
                    body=path.read_text(encoding="utf-8"))

    @app.get("/paper/{paper_id}", response_class=HTMLResponse)
    def paper_page(request: Request, paper_id: int, tab: str = "summary"):
        row = db.get(paper_id)
        if row is None:
            raise HTTPException(404)
        docs = {}
        if row["status"] == "done":
            out = _safe_output_dir(cfg, row)
            for key, fname in (("summary", "summary.md"), ("sections", "sections.md")):
                if (out / fname).exists():
                    docs[key] = strip_title_block((out / fname).read_text(encoding="utf-8"))
        verification = None
        if row["output_dir"] and (Path(row["output_dir"]) / "verification.json").exists():
            verification = json.loads((Path(row["output_dir"]) / "verification.json").read_text(encoding="utf-8"))
        return page(request, "paper.html", row=row, docs=docs, tab=tab if tab in docs else "summary",
                    authors=json.loads(row["authors"]) if row["authors"] else [], verification=verification)

    @app.get("/paper/{paper_id}/pdf")
    def paper_pdf(request: Request, paper_id: int):
        require_user(request)
        row = db.get(paper_id)
        if row is None:
            raise HTTPException(404)
        if row["status"] == "done":
            path = _safe_output_dir(cfg, row) / "paper.pdf"
        else:
            path = Path(row["inbox_path"] or "")
            roots = (cfg.paths.inbox.resolve(), cfg.paths.failed.resolve())
            if not any(path.resolve().is_relative_to(r) for r in roots):
                raise HTTPException(404)
        if not path.exists():
            raise HTTPException(404, "PDF が見つかりません")
        return FileResponse(path, media_type="application/pdf", filename=row["source_name"],
                            content_disposition_type="inline")

    @app.get("/api/paper/{paper_id}/evidence")
    def evidence(request: Request, paper_id: int, ids: str):
        require_user(request)
        row = db.get(paper_id)
        if row is None or row["status"] != "done":
            raise HTTPException(404)
        path = _safe_output_dir(cfg, row) / "evidence.json"
        if not path.exists():
            raise HTTPException(404, "この要約には根拠段落のデータがありません")
        index = json.loads(path.read_text(encoding="utf-8"))
        items = [{"id": pid, **index[pid]} for pid in CITE_ID_RE.findall(ids)[:10] if pid in index]
        return {"ok": True, "items": items}

    # -- 失敗・再処理 --
    @app.get("/failed", response_class=HTMLResponse)
    def failed_page(request: Request):
        rows = db.by_status(("failed", "duplicate", "missing"), 100)
        errors = {}
        for r in rows:
            p = Path(r["inbox_path"] or "")
            et = p.with_name(p.name + ".error.txt")
            if r["status"] == "failed" and et.exists():
                errors[r["id"]] = et.read_text(encoding="utf-8")[:3000]
        return page(request, "failed.html", rows=rows, errors=errors)

    @app.post("/api/paper/{paper_id}/{action}")
    def paper_action(request: Request, paper_id: int, action: str):
        user = require_user(request)
        require_csrf(request, request.headers.get("x-csrf-token"))
        try:
            if action == "retry":
                ops.retry(cfg, db, paper_id)
                return {"ok": True, "message": "再試行の待ち行列に入れました"}
            if action == "reprocess":
                pid = ops.reprocess(cfg, db, paper_id, user)
                return {"ok": True, "message": f"再処理を登録しました (#{pid})"}
        except ops.OpError as e:
            raise HTTPException(400, str(e)) from e
        raise HTTPException(404)

    return app
