import json
import re

import pytest
from fastapi.testclient import TestClient

from paper_summarizer.config import Config, Paths, ensure_dirs
from paper_summarizer.db import DB
from paper_summarizer.web import auth
from paper_summarizer.web.app import create_app, render_markdown

PDF = b"%PDF-1.4\n%test\n"


@pytest.fixture
def client(tmp_path):
    paths = Paths(root=tmp_path / "papers", db=tmp_path / "papers/.state/db.sqlite3",
                  log_dir=tmp_path / "logs", work_dir=tmp_path / "work")
    cfg = Config(paths=paths, scan={"min_age_seconds": 120}, ollama={"url": "http://127.0.0.1:9", "model": "m"},
                 extract={}, summarize={}, notify={}, secrets={}, web={"https": False})
    ensure_dirs(cfg)
    auth.set_password(paths.db.parent, "ito", "password123")
    c = TestClient(create_app(cfg))
    c.cfg = cfg
    return c


def _csrf(c, path="/login"):
    return re.search(r'name="csrf(?:-token)?" (?:value|content)="(\w+)"', c.get(path).text).group(1)


def _login(c):
    r = c.post("/login", data={"username": "ito", "password": "password123", "csrf": _csrf(c)},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    return re.search(r'name="csrf-token" content="(\w+)"', c.get("/").text).group(1)


def test_requires_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/fragment/status").status_code == 401
    assert client.post("/api/run").status_code == 401


def test_bad_password_and_csrf(client):
    r = client.post("/login", data={"username": "ito", "password": "wrong", "csrf": _csrf(client)},
                    follow_redirects=False)
    assert r.headers["location"] == "/login?error=1"
    r = client.post("/login", data={"username": "ito", "password": "password123", "csrf": "bad"})
    assert r.status_code == 403


def test_pages_render(client):
    _login(client)
    for path in ("/", "/upload", "/library", "/failed", "/fragment/status"):
        assert client.get(path).status_code == 200, path


def test_upload(client):
    token = _login(client)
    h = {"X-CSRF-Token": token}
    r = client.post("/api/upload", data={"project": "proj-a"}, headers=h,
                    files=[("files", ("../../evil name.pdf", PDF, "application/pdf"))])
    res = r.json()["results"][0]
    assert res["ok"] and res["name"] == "evil name.pdf"
    saved = client.cfg.paths.inbox / "proj-a" / "evil name.pdf"
    assert saved.read_bytes() == PDF
    assert not list((client.cfg.paths.inbox / "proj-a").glob(".*"))  # 一時ファイルが残らない
    db = DB(client.cfg.paths.db)
    assert db.conn.execute("SELECT user FROM uploads").fetchone()["user"] == "ito"

    # 同じ内容 (処理待ち扱いは scan 後) / PDF 以外 / 不正なプロジェクト名
    r = client.post("/api/upload", data={"project": "proj-a"}, headers=h,
                    files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert not r.json()["results"][0]["ok"]
    r = client.post("/api/upload", data={"project": "../etc"}, headers=h,
                    files=[("files", ("a.pdf", PDF, "application/pdf"))])
    assert r.status_code == 400
    r = client.post("/api/upload", data={"project": "proj-a"}, files=[("files", ("a.pdf", PDF, "application/pdf"))])
    assert r.status_code == 403  # CSRF トークンなし


def test_scan_records_uploader(client):
    from paper_summarizer.pipeline import RunReport, scan
    token = _login(client)
    client.post("/api/upload", data={"project": "p"}, headers={"X-CSRF-Token": token},
                files=[("files", ("a.pdf", PDF, "application/pdf"))])
    db = DB(client.cfg.paths.db)
    scan(client.cfg, db, RunReport())  # min_age_seconds=120 でも投稿直後に登録される
    (row,) = db.pending()
    assert row["submitted_by"] == "ito"
    r = client.post("/api/upload", data={"project": "p"}, headers={"X-CSRF-Token": token},
                    files=[("files", ("b.pdf", PDF, "application/pdf"))])
    assert "処理待ち" in r.json()["results"][0]["message"]


def test_paper_and_evidence(client):
    _login(client)
    cfg = client.cfg
    out = cfg.paths.library / "p" / "2020_X_t"
    out.mkdir(parents=True)
    (out / "summary.md").write_text("# T\n\n## 1. どんなもの？\n- $a_b$ を使う。 [§1-p1]\n")
    (out / "evidence.json").write_text(json.dumps({"§1-p1": {"section": "1", "section_title": "Intro",
                                                             "kind": "text", "text": "原文です"}}))
    (out / "paper.pdf").write_bytes(PDF)
    db = DB(cfg.paths.db)
    pid = db.add(sha256="s", project="p", source_name="a.pdf", inbox_path="", status="done")
    db.update(pid, output_dir=str(out), title="T")
    page = client.get(f"/paper/{pid}").text
    assert 'class="cite" data-ids="§1-p1"' in page and "$a_b$" in page
    assert client.get(f"/api/paper/{pid}/evidence", params={"ids": "§1-p1"}).json()["items"][0]["text"] == "原文です"
    assert client.get(f"/paper/{pid}/pdf").content == PDF
    db.update(pid, output_dir="/etc")  # library 外は拒否
    assert client.get(f"/paper/{pid}/pdf").status_code == 404


def test_render_markdown_keeps_math_and_links_citations():
    h = render_markdown("- 式 $x_1 * y_2$ と $$\\frac{a}{b}$$ を使う。 [§2.1-p3, §3-p1]\n- 普通の [リンク](http://x)")
    assert "$x_1 * y_2$" in h and "<em>" not in h
    assert 'data-ids="§2.1-p3,§3-p1"' in h and '<a href="http://x">' in h
