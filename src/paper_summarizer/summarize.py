"""分割要約 → 統合 → 自己検証ループ。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .extract import Extracted, split_chunks
from .llm import LLMError, OllamaClient
from .prompts import load, render

log = logging.getLogger(__name__)

SECTION_TITLES = [
    "1. どんなもの？",
    "2. 先行研究を比べてどこがすごい？",
    "3. 技術や手法の肝はどこ？",
    "4. どうやって有効だと検証した？",
    "5. 議論はある？",
    "6. 次に読むべき論文は？",
]


@dataclass
class SummaryResult:
    markdown: str
    n_chunks: int
    rounds: list[dict] = field(default_factory=list)  # 検証ラウンドごとの指摘
    converged: bool = False


def summarize(ext: Extracted, title: str, llm: OllamaClient, cfg: dict) -> SummaryResult:
    chunk_chars = int(cfg.get("chunk_chars", 24000))
    limit = int(cfg.get("section_char_limit", 500))
    max_rounds = int(cfg.get("verify_rounds", 3))
    base_prompt = load("paper_summary").strip()
    fmt = render("format", section_char_limit=limit)

    chunks = split_chunks(ext.markdown, chunk_chars)
    total = len(chunks)
    log.info("分割: %d チャンク (最大 %d 文字)", total, chunk_chars)

    # 1) 読書メモ (map)
    notes = []
    for i, chunk in enumerate(chunks, 1):
        res = llm.chat(render("chunk_notes", part=i, total=total, outline=ext.outline, chunk=chunk),
                       stage=f"notes:{i}/{total}")
        notes.append(f"### パート {i}/{total}\n{res.content.strip()}")

    # 2) 統合 (reduce)
    res = llm.chat(render("merge", base_prompt=base_prompt, title=title, abstract=ext.abstract,
                          notes="\n\n".join(notes), format=fmt), stage="merge")
    summary = normalize(res.content)

    # 3) 自己検証ループ
    #    全文が収まる場合は全文と照合する (分割照合は他パートの情報を「誤り」と誤判定しやすい)。
    full_max = int(cfg.get("verify_full_max_chars", 200000))
    verify_ctx = int(cfg.get("verify_num_ctx", 131072))
    if len(ext.markdown) <= full_max:
        targets = [("", "", ext.markdown)]
        log.info("検証: 全文照合 (%d 文字)", len(ext.markdown))
    else:
        note = ("注意: 原文は長いため分割されており、以下はその一部です。このパートに書かれていないだけの記述は"
                "他のパートに根拠がある可能性があるため「誤り」にしないこと。このパートと明確に矛盾する場合のみ誤りとする。")
        targets = [(note, f" (パート {i}/{total})", c) for i, c in enumerate(chunks, 1)]
        log.info("検証: 分割照合 (%d パート)", total)

    result = SummaryResult(markdown=summary, n_chunks=total)
    for rnd in range(1, max_rounds + 1):
        issues = format_issues(summary, limit)
        for i, (note, label, text) in enumerate(targets, 1):
            try:
                data = llm.chat_json(
                    render("verify", scope_note=note, scope_label=label, summary=summary, chunk=text,
                           section_char_limit=limit),
                    stage=f"verify:r{rnd}:{i}/{len(targets)}", num_ctx=verify_ctx)
            except LLMError as e:
                log.warning("検証応答を解釈できませんでした (round %d part %d): %r", rnd, i, e)
                continue
            for it in data.get("issues", []) or []:
                if isinstance(it, dict) and str(it.get("confidence", "high")).lower() != "low":
                    it["part"] = i
                    issues.append(it)
        result.rounds.append({"round": rnd, "issues": issues})
        log.info("検証 round %d: 指摘 %d 件", rnd, len(issues))
        if not issues:
            result.converged = True
            break
        res = llm.chat(render("revise", base_prompt=base_prompt, summary=summary,
                              issues=_format_issue_list(issues), format=fmt), stage=f"revise:r{rnd}")
        summary = normalize(res.content)

    if not result.converged:
        remaining = format_issues(summary, limit)
        if remaining:
            log.warning("検証ループ終了時点で形式上の問題が残っています: %s", remaining)
        else:
            log.info("検証ラウンド上限に到達 (最終修正後の内容照合は未実施)")
    result.markdown = summary
    return result


# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """前置きやコードフェンスを除去し、最初の見出しから始まるようにする。"""
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text.strip())
    m = re.search(r"(?m)^##\s*1\.", text)
    return (text[m.start():] if m else text).strip() + "\n"


def split_sections(summary: str) -> dict[int, str]:
    sections: dict[int, str] = {}
    for m in re.finditer(r"(?ms)^##\s*(\d)\.[^\n]*\n(.*?)(?=^##\s*\d\.|\Z)", summary):
        sections[int(m.group(1))] = m.group(2).strip()
    return sections


def body_chars(body: str) -> int:
    """箇条書き記号・インデント・改行を除いた文字数。"""
    lines = [re.sub(r"^\s*[-*+]\s*", "", ln).strip() for ln in body.splitlines()]
    return sum(len(ln) for ln in lines)


def format_issues(summary: str, limit: int) -> list[dict]:
    issues: list[dict] = []
    sections = split_sections(summary)
    for n, title in enumerate(SECTION_TITLES, 1):
        if n not in sections:
            issues.append({"type": "形式", "section": n, "fix": f"見出し「## {title}」が欠けている。追加すること"})
            continue
        body = sections[n]
        chars = body_chars(body)
        if chars > limit:
            issues.append({"type": "形式", "section": n,
                           "fix": f"本文が {chars} 文字で上限 {limit} 文字を超過。重要度の低い記述を削って {limit} 文字以内にすること"})
        if not any(re.match(r"^\s*[-*+]\s", ln) for ln in body.splitlines()):
            issues.append({"type": "形式", "section": n, "fix": "階層付き箇条書きになっていない"})
    return issues


def _format_issue_list(issues: list[dict]) -> str:
    lines = []
    for k, it in enumerate(issues, 1):
        parts = [f"{k}. [{it.get('type', '?')}] 項目{it.get('section', '?')}"]
        if it.get("summary_text"):
            parts.append(f"該当箇所: {it['summary_text']}")
        if it.get("evidence"):
            parts.append(f"原文: {it['evidence']}")
        if it.get("fix"):
            parts.append(f"修正方針: {it['fix']}")
        lines.append(" / ".join(parts))
    return "\n".join(lines)
