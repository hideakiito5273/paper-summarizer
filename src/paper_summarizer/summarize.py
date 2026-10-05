"""分割要約 → 統合 → 自己検証ループ。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .extract import Extracted, split_chunks
from .llm import LLMError, OllamaClient
from .prompts import load, render
from .summarize_util import run_parallel

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
    def note(i: int, chunk: str) -> str:
        res = llm.chat(render("chunk_notes", part=i, total=total, outline=ext.outline, chunk=chunk),
                       stage=f"notes:{i}/{total}")
        return f"### パート {i}/{total}\n{res.content.strip()}"

    notes = run_parallel([lambda i=i, c=c: note(i, c) for i, c in enumerate(chunks, 1)], llm.parallel)

    # 2) 統合 (reduce)
    res = llm.chat(render("merge", base_prompt=base_prompt, title=title, abstract=ext.abstract,
                          notes="\n\n".join(notes), format=fmt), stage="merge")
    summary = shorten_if_needed(normalize(res.content), llm, limit, fmt, stage="shorten:merge")

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
        failed_parts: list[int] = []
        for i, (note, label, text) in enumerate(targets, 1):
            try:
                data = llm.chat_json(
                    render("verify", scope_note=note, scope_label=label, summary=summary, chunk=text,
                           section_char_limit=limit),
                    stage=f"verify:r{rnd}:{i}/{len(targets)}", num_ctx=verify_ctx)
            except LLMError as e:
                log.warning("照合に失敗 (round %d part %d): %s", rnd, i, e)
                failed_parts.append(i)
                continue
            for it in data.get("issues", []) or []:
                if isinstance(it, dict) and str(it.get("confidence", "high")).lower() != "low":
                    it["part"] = i
                    issues.append(it)
        rec = {"round": rnd, "issues": issues}
        if failed_parts:
            rec["verify_failed_parts"] = failed_parts
        result.rounds.append(rec)
        log.info("検証 round %d: 指摘 %d 件%s", rnd, len(issues),
                 f" (照合失敗 {len(failed_parts)}/{len(targets)} パート)" if failed_parts else "")
        if len(failed_parts) == len(targets):
            # 照合が 1 つも成功していない = 未検証。「指摘なし」と誤認しないよう収束扱いにしない
            log.warning("照合がすべて失敗したため検証を打ち切ります (round %d)", rnd)
            if not issues:
                break
        elif not issues and not failed_parts:
            result.converged = True
            break
        elif not issues:  # 一部パートのみ照合成功で指摘なし → 再照合しても同じ入力なので終了
            break
        try:
            res = llm.chat(render("revise", base_prompt=base_prompt, summary=summary,
                                  issues=_format_issue_list(issues), format=fmt), stage=f"revise:r{rnd}")
        except LLMError as e:  # 修正に失敗しても、それまでの要約は有効なので残す
            log.warning("修正 round %d に失敗、直前の要約を採用して検証を終了: %s", rnd, e)
            result.rounds[-1]["revise_failed"] = str(e)
            break
        revised = normalize(res.content)
        if len(split_sections(revised)) < len(split_sections(summary)):
            log.warning("修正版で見出しが減ったため採用しません (round %d)", rnd)
            result.rounds[-1]["revise_rejected"] = "見出しが減少"
            break
        summary = shorten_if_needed(revised, llm, limit, fmt, stage=f"shorten:r{rnd}")

    if not result.converged:
        remaining = format_issues(summary, limit)
        if remaining:
            log.warning("検証ループ終了時点で形式上の問題が残っています: %s", remaining)
        else:
            log.info("検証ラウンド上限に到達 (最終修正後の内容照合は未実施)")
    result.markdown = summary
    return result


# ---------------------------------------------------------------------------
def shorten_if_needed(summary: str, llm: OllamaClient, limit: int, fmt: str, *, stage: str) -> str:
    """文字数超過の項目だけを推論の浅い専用工程で短くする (照合を内容の確認に集中させるため)。"""
    sections = split_sections(summary)
    over = {n: body_chars(b) for n, b in sections.items() if body_chars(b) > limit}
    if not over:
        return summary
    # 上限の 9 割を目標に、何割削ればよいかを渡す (モデルに文字数を数えさせない)
    targets = "\n".join(
        f"- 項目 {n} ({SECTION_TITLES[n - 1]}): 約 {c} 文字 → {int(limit * 0.9)} 文字程度に "
        f"(約 {max(1, round((1 - limit * 0.9 / c) * 10))} 割削る)"
        for n, c in sorted(over.items()))
    log.info("短縮: %s", ", ".join(f"項目{n}={c}字" for n, c in sorted(over.items())))
    try:
        res = llm.chat(render("shorten", section_char_limit=limit, targets=targets, summary=summary, format=fmt),
                       stage=stage)
    except LLMError as e:
        log.warning("短縮に失敗、元の要約のまま照合に進みます: %s", e)
        return summary
    shortened = normalize(res.content)
    if len(split_sections(shortened)) < len(sections):
        log.warning("短縮版で見出しが減ったため採用しません")
        return summary
    return shortened


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
