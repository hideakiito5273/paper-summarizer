import os
import time
from pathlib import Path

import pytest

from paper_summarizer.config import Config, Paths, ensure_dirs
from paper_summarizer.db import DB
from paper_summarizer.extract import _replace_in_order, split_chunks
from paper_summarizer.notify import build_message
from paper_summarizer.output import paper_dir_name
from paper_summarizer.pipeline import RunReport, project_of, scan
from paper_summarizer.prompts import render
from paper_summarizer.summarize import body_chars, format_issues, normalize, split_sections

GOOD = """## 論旨の流れ
- 問題: 問題がある。 [§1-p1]
- 手法: 手法を使う。 [§2-p1]
- 結果: 結果が出た。 [§3-p1]
## 1. どんなもの？
- A である。 [§1-p1]
  - B である。 [§1-p2]
## 2. 先行研究を比べてどこがすごい？
- C である。 [§1-p1]
## 3. 技術や手法の肝はどこ？
- D である。 [§2-p1]
## 4. どうやって有効だと検証した？
- E である。 [§3-p1]
## 5. 議論はある？
- F である。 [§3-p1]
## 6. 次に読むべき論文は？
- G (2000) 「H」: I をした。 [§1-p2]
"""
IDS = {"§1-p1", "§1-p2", "§2-p1", "§3-p1"}


def test_split_chunks_respects_limit():
    md = "\n".join(f"## S{i}\n" + ("word " * 300) for i in range(20))
    chunks = split_chunks(md, 4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks).replace("\n", "") .count("word") == md.count("word")


def test_split_chunks_long_paragraph():
    chunks = split_chunks("x" * 10000, 3000)
    assert all(len(c) <= 3000 for c in chunks) and sum(map(len, chunks)) == 10000


def test_format_ok_and_limits():
    assert format_issues(GOOD, 500, IDS) == []
    long = GOOD.replace("- A である。", "- " + "あ" * 501)
    issues = format_issues(long, 500, IDS)
    assert len(issues) == 1 and issues[0]["section"] == 1
    missing = GOOD.split("## 6.")[0]
    assert any(i["section"] == 6 for i in format_issues(missing, 500, IDS))


def test_citations_not_counted_and_checked():
    assert body_chars("- ab [§1-p1, §2-p3]\n  - cd [§1-p2]") == 4
    no_cite = GOOD.replace("- C である。 [§1-p1]", "- C である。")
    assert any(i["type"] == "根拠不備" and i["section"] == 2 for i in format_issues(no_cite, 500, IDS))
    bad_id = GOOD.replace("[§2-p1]", "[§9-p9]")
    assert any("存在しない段落 ID" in i["fix"] for i in format_issues(bad_id, 500, IDS))


def test_flow_required():
    no_flow = GOOD[GOOD.index("## 1."):]
    assert any(i["section"] == "流れ" for i in format_issues(no_flow, 500, IDS))


def test_normalize_strips_preamble():
    out = normalize("はい、要約です。\n```markdown\n" + GOOD + "```")
    assert out.startswith("## 論旨の流れ") and len(split_sections(out)) == 7


def test_body_chars_ignores_bullets():
    assert body_chars("- ab\n  - cd") == 4


def test_replace_in_order():
    assert _replace_in_order("a<m>b<m>c", "<m>", ["1", "2"]) == "a1b2c"


def test_render_requires_vars():
    with pytest.raises(KeyError):
        render("figure")
    assert "キャプション: X" in render("figure", caption="X")


def test_paper_dir_name():
    meta = {"year": 2024, "authors": ["Smith, John"], "short_title": "Attention is all"}
    assert paper_dir_name(meta) == "2024_Smith_Attention-is-all"
    assert paper_dir_name({}) == "XXXX_Unknown_untitled"
    assert paper_dir_name({"year": 2011, "authors": ["Ebru Bozdağ"], "short_title": "misfit"}) == "2011_Bozdag_misfit"


# ---- scan / 状態遷移 ---------------------------------------------------------
@pytest.fixture
def env(tmp_path):
    paths = Paths(root=tmp_path / "papers", db=tmp_path / "state" / "db.sqlite3",
                  log_dir=tmp_path / "logs", work_dir=tmp_path / "work")
    cfg = Config(paths=paths, scan={"min_age_seconds": 0}, ollama={"model": "m"}, extract={},
                 summarize={}, notify={}, secrets={}, web={})
    ensure_dirs(cfg)
    return cfg, DB(paths.db)


def _pdf(path: Path, content: bytes = b"%PDF-1.4 test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    old = time.time() - 600
    os.utime(path, (old, old))
    return path


def test_scan_registers_and_ignores_temp(env):
    cfg, db = env
    _pdf(cfg.paths.inbox / "projA" / "a.pdf")
    _pdf(cfg.paths.inbox / "root.pdf", b"other")
    _pdf(cfg.paths.inbox / "projA" / ".syncthing.b.pdf.tmp")
    _pdf(cfg.paths.inbox / "projA" / ".stversions" / "old.pdf", b"old")
    scan(cfg, db, RunReport())
    rows = db.pending()
    assert {(r["project"], r["source_name"]) for r in rows} == {("projA", "a.pdf"), ("_unsorted", "root.pdf")}
    scan(cfg, db, RunReport())  # 2 回目は増えない
    assert len(db.pending()) == 2


def test_scan_skips_fresh_files(env):
    cfg, db = env
    cfg.scan["min_age_seconds"] = 120
    p = cfg.paths.inbox / "projA" / "new.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF")
    scan(cfg, db, RunReport())
    assert db.pending() == []


def test_duplicate_same_project_goes_to_failed(env):
    cfg, db = env
    out = cfg.paths.library / "projA" / "x"
    out.mkdir(parents=True)
    pid = db.add(sha256="h", project="projA", source_name="a.pdf", inbox_path="", status="done")
    import hashlib
    content = b"%PDF dup"
    db.update(pid, sha256=hashlib.sha256(content).hexdigest(), output_dir=str(out))
    _pdf(cfg.paths.inbox / "projA" / "copy.pdf", content)
    report = RunReport()
    scan(cfg, db, report)
    assert len(report.duplicates) == 1
    assert (cfg.paths.failed / "projA" / "copy.pdf").exists()
    assert (cfg.paths.failed / "projA" / "copy.pdf.error.txt").exists()


def test_replacement_links_previous(env):
    cfg, db = env
    old = db.add(sha256="old", project="projA", source_name="a.pdf", inbox_path="", status="done")
    _pdf(cfg.paths.inbox / "projA" / "a.pdf", b"%PDF new version")
    scan(cfg, db, RunReport())
    (row,) = db.pending()
    assert row["replaces"] == old


def test_missing_file_marked(env):
    cfg, db = env
    p = _pdf(cfg.paths.inbox / "projA" / "a.pdf")
    scan(cfg, db, RunReport())
    p.unlink()
    scan(cfg, db, RunReport())
    assert db.pending() == []


def test_project_of(tmp_path):
    assert project_of(tmp_path / "p" / "a.pdf", tmp_path) == "p"
    assert project_of(tmp_path / "a.pdf", tmp_path) == "_unsorted"


def test_mail_has_titles_not_content():
    r = RunReport(done=[{"id": 1, "title": "Paper T", "project": "p", "minutes": 3.0}],
                  failed=[{"name": "b.pdf", "project": "p", "error": "ValueError: x", "final": True}])
    subject, body = build_message(r, True, "rid")
    assert "完了 1 件" in subject and "失敗 1 件" in subject
    assert "Paper T" in body and "b.pdf" in body
    _, body2 = build_message(r, False, "rid")
    assert "Paper T" not in body2


def test_daily_report_once_per_day(env, monkeypatch):
    from datetime import datetime
    from paper_summarizer import notify
    cfg, db = env
    sent = []
    monkeypatch.setattr(notify, "send", lambda c, s, b: sent.append((s, b)) or True)
    monkeypatch.setattr(notify, "build_status", lambda c, d: "■ 稼働状況")
    assert not notify.daily_report_due(cfg, datetime(2026, 10, 7, 0, 0))   # 6 時前は送らない
    assert notify.daily_report_due(cfg, datetime(2026, 10, 7, 6, 0))
    notify.notify_report(cfg, RunReport(), "rid", db)                      # 空でも 1 日 1 回は送る
    notify.notify_report(cfg, RunReport(), "rid", db)                      # 同じ日の 2 回目は送らない
    assert len(sent) == 1 and "稼働報告" in sent[0][0] and "■ 稼働状況" in sent[0][1]
    r = RunReport(done=[{"id": 1, "title": "T", "project": "p", "minutes": 1.0}])
    notify.notify_report(cfg, r, "rid", db)                                # 処理があれば都度送る (稼働状況なし)
    assert len(sent) == 2 and "完了 1 件" in sent[1][0] and "■ 稼働状況" not in sent[1][1]
