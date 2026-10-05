"""プロジェクト単位の文献レビュー (手動実行)。"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from .config import Config
from .db import DB
from .llm import OllamaClient
from .output import update_project_readme
from .prompts import render
from .summarize import strip_citations

log = logging.getLogger(__name__)


def _paper_block(row) -> str:
    authors = json.loads(row["authors"]) if row["authors"] else []
    head = f"### [{(authors[0] if authors else '不明')} {row['year'] or ''}] {row['title'] or row['source_name']}"
    text = (Path(row["output_dir"]) / "summary.md").read_text(encoding="utf-8")
    m = re.search(r"(?m)^##\s*(論旨の流れ|1\.)", text)  # 書誌表を除いた本文のみ
    body = text[m.start():] if m else text
    return f"{head}\n{strip_citations(body)}"  # 段落 ID は論文ごとの番号なので横断レビューでは除く


def run_review(cfg: Config, db: DB, llm: OllamaClient, project: str) -> Path:
    rows = db.done_in_project(project)
    if not rows:
        raise SystemExit(f"プロジェクト {project!r} に処理済みの論文がありません")
    blocks = [_paper_block(r) for r in rows]
    budget = int(cfg.summarize.get("review_max_input_chars", 60000))
    llm.check([cfg.ollama["model"]])
    log.info("レビュー作成開始: %s (%d 本)", project, len(rows))

    groups = _group(blocks, budget)
    if len(groups) == 1:
        body = llm.chat(render("project_review", project=project, n=len(rows), summaries=groups[0]),
                        stage="review").content
    else:
        # 要約の総量が多い場合: グループごとに中間レビュー → それらを統合
        partials = []
        for i, g in enumerate(groups, 1):
            res = llm.chat(render("project_review", project=f"{project} (部分 {i}/{len(groups)})",
                                  n=g.count("\n### ") + 1, summaries=g), stage=f"review:part{i}")
            partials.append(f"### 部分レビュー {i}\n{res.content}")
        body = llm.chat(render("project_review", project=project, n=len(rows),
                               summaries="以下は論文群を分割して作成した部分レビューです。\n\n"
                                         + "\n\n".join(partials)),
                        stage="review:merge").content

    out_dir = cfg.paths.reviews / project
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"review_{datetime.now():%Y-%m-%d}.md"
    if path.exists():
        path = out_dir / f"review_{datetime.now():%Y-%m-%d_%H%M}.md"
    refs = "\n".join(f"- {b.splitlines()[0][4:]}" for b in blocks)
    path.write_text(
        f"# 文献レビュー: {project}\n\n- 作成: {datetime.now():%Y-%m-%d %H:%M}\n- 対象: {len(rows)} 本\n"
        f"- モデル: {cfg.ollama['model']}\n\n---\n\n{body.strip()}\n\n---\n\n## 対象論文\n\n{refs}\n",
        encoding="utf-8",
    )
    update_project_readme(db, cfg.paths.reviews, cfg.paths.library, project)
    log.info("レビュー作成完了: %s", path)
    return path


def _group(blocks: list[str], budget: int) -> list[str]:
    groups, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > budget:
            groups.append(cur)
            cur = ""
        cur = f"{cur}\n\n{b}" if cur else b
    groups.append(cur)
    return groups
