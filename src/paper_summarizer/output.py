"""library/ への成果物出力とプロジェクト README (文献一覧) の生成。"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from .db import DB
from .summarize import split_sections

log = logging.getLogger(__name__)

HISTORY_DIR = "_history"


def slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\-]+", "-", text, flags=re.ASCII).strip("-")
    return re.sub(r"-{2,}", "-", text)[:max_len].strip("-") or "untitled"


def last_name(author: str) -> str:
    author = author.strip()
    if "," in author:
        return author.split(",")[0].strip()
    return author.split()[-1] if author.split() else "Unknown"


def paper_dir_name(meta: dict) -> str:
    year = meta.get("year") or "XXXX"
    authors = meta.get("authors") or []
    first = slug(last_name(authors[0]), 30) if authors else "Unknown"
    short = slug(meta.get("short_title") or meta.get("title") or "untitled", 50)
    return f"{year}_{first}_{short}"


def unique_dir(parent: Path, name: str) -> Path:
    d = parent / name
    k = 2
    while d.exists():
        d = parent / f"{name}_{k}"
        k += 1
    return d


def archive_previous(out_dir: Path) -> None:
    """差し替え時: 既存の成果物を _history/<日時>/ に退避する。"""
    files = [p for p in out_dir.iterdir() if p.name != HISTORY_DIR]
    if not files:
        return
    dest = out_dir / HISTORY_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True)
    for p in files:
        shutil.move(str(p), dest / p.name)
    log.info("旧版を退避: %s", dest)


def render_summary(meta: dict, summary_md: str, verification: dict) -> str:
    authors = ", ".join(meta.get("authors") or []) or "不明"
    rows = [
        ("著者", authors),
        ("年", meta.get("year") or "不明"),
        ("掲載", meta.get("venue") or "不明"),
        ("DOI", f"[{meta['doi']}](https://doi.org/{meta['doi']})" if meta.get("doi") else "不明"),
        ("プロジェクト", meta["project"]),
        ("元ファイル", meta["source_name"]),
        ("処理日時", meta["processed_at"]),
        ("モデル", meta["model"]),
        ("自己検証", verification_label(verification)),
    ]
    table = "| 項目 | 内容 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"# {meta.get('title') or meta['source_name']}\n\n{table}\n\n---\n\n{summary_md.strip()}\n"


def verification_label(v: dict) -> str:
    n = len(v.get("rounds", []))
    if v.get("converged"):
        return f"{n} ラウンドで指摘なし"
    last = v["rounds"][-1]["issues"] if v.get("rounds") else []
    return f"{n} ラウンド実施 (最終ラウンドの指摘 {len(last)} 件を修正済み・再照合なし)"


def write_outputs(out_dir: Path, pdf: Path, meta: dict, summary_md: str, verification: dict,
                  extracted_md: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(render_summary(meta, summary_md, verification), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    shutil.copy2(extracted_md, out_dir / "extracted.md")
    shutil.move(str(pdf), out_dir / "paper.pdf")


def first_point(summary_path: Path, max_len: int = 90) -> str:
    try:
        sections = split_sections(summary_path.read_text(encoding="utf-8"))
    except OSError:
        return ""
    for line in sections.get(1, "").splitlines():
        text = re.sub(r"^\s*[-*+]\s*", "", line).strip()
        if text:
            return text if len(text) <= max_len else text[: max_len - 1] + "…"
    return ""


def update_project_readme(db: DB, reviews_dir: Path, library_dir: Path, project: str) -> Path:
    rows = db.done_in_project(project)
    out_dir = reviews_dir / project
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.md"

    lines = [
        f"# 文献一覧: {project}",
        "",
        f"- 論文数: {len(rows)}",
        f"- 更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 全体レビューの作成: `paper-summarizer review {project}`",
        "",
        "| 年 | 第一著者 | タイトル | 要旨 (1. どんなもの？ の冒頭) | 要約 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        authors = json.loads(r["authors"]) if r["authors"] else []
        out = Path(r["output_dir"])
        rel = Path("..", "..", out.relative_to(library_dir.parent), "summary.md").as_posix()
        title = (r["title"] or r["source_name"]).replace("|", "\\|")
        gist = first_point(out / "summary.md").replace("|", "\\|")
        lines.append(
            f"| {r['year'] or ''} | {last_name(authors[0]) if authors else ''} | {title} | {gist} | [summary]({rel}) |"
        )

    reviews = sorted(out_dir.glob("review_*.md"), reverse=True)
    if reviews:
        lines += ["", "## 全体レビュー", ""] + [f"- [{p.stem}]({p.name})" for p in reviews]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme
