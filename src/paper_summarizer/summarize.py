"""構造化読解: 節ごとの精読 (段落メモ + 節要約) → 統合 → 短縮 → 根拠段落との照合ループ。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .extract import Extracted
from .llm import LLMError, OllamaClient
from .prompts import load, render
from .structure import Paragraph, Section, paragraph_index

log = logging.getLogger(__name__)

FLOW_TITLE = "論旨の流れ"
SECTION_TITLES = [
    "1. どんなもの？",
    "2. 先行研究を比べてどこがすごい？",
    "3. 技術や手法の肝はどこ？",
    "4. どうやって有効だと検証した？",
    "5. 議論はある？",
    "6. 次に読むべき論文は？",
]
FLOW = "flow"  # split_sections のキー
CITE_RE = re.compile(r"\[(§[^\]]+)\]")
CITE_ID_RE = re.compile(r"§[\w.]+-p\d+")


@dataclass
class Readings:
    """節ごとの精読結果。"""
    sections: dict[str, dict] = field(default_factory=dict)  # 節 ID → {title, role, summary}
    notes: dict[str, str] = field(default_factory=dict)      # 段落 ID → 要点

    def render(self, all_sections: list[Section]) -> str:
        out = []
        for sec in all_sections:
            info = self.sections.get(sec.id)
            if info is None:
                continue
            out.append(f"## §{sec.id} {sec.title} 〔役割: {info.get('role', '?')}〕\n節の要約: {info.get('summary', '')}")
            out += [f"- [{p.id}] {self.notes.get(p.id, '')}" for p in sec.paragraphs if p.id in self.notes]
        return "\n".join(out)

    def section_summaries(self, all_sections: list[Section]) -> str:
        return "\n".join(f"- §{s.id} {s.title} 〔{self.sections[s.id].get('role', '?')}〕: "
                         f"{self.sections[s.id].get('summary', '')}"
                         for s in all_sections if s.id in self.sections)

    def notes_index(self) -> str:
        return "\n".join(f"- {pid}: {note}" for pid, note in self.notes.items())


@dataclass
class SummaryResult:
    markdown: str
    n_chunks: int  # 精読の呼び出し単位の数
    readings: Readings = field(default_factory=Readings)
    rounds: list[dict] = field(default_factory=list)  # 検証ラウンドごとの指摘
    converged: bool = False


# ---------------------------------------------------------------------------
def summarize(ext: Extracted, title: str, llm: OllamaClient, cfg: dict) -> SummaryResult:
    limit = int(cfg.get("section_char_limit", 1000))
    max_rounds = int(cfg.get("verify_rounds", 3))
    fmt = render("format", section_char_limit=limit)
    paragraphs = paragraph_index(ext.sections)

    # 1) 節ごとの精読
    units = reading_units(ext.sections, int(cfg.get("read_chars", 12000)))
    log.info("精読: %d 単位 (節 %d, 段落 %d)", len(units), sum(1 for s in ext.sections if s.readable),
             sum(len(s.paragraphs) for s in ext.sections if s.readable))
    readings = read_sections(units, ext, title, llm)

    # 2) 統合
    res = llm.chat(render("synthesize", base_prompt=load("paper_summary").strip(), title=title,
                          abstract=ext.abstract, readings=readings.render(ext.sections),
                          references=ext.references[: int(cfg.get("references_chars", 15000))] or "(抽出できず)",
                          format=fmt), stage="merge")
    summary = shorten_if_needed(normalize(res.content), llm, limit, fmt, stage="shorten:merge")

    # 3) 根拠段落との照合ループ
    result = SummaryResult(markdown=summary, n_chunks=len(units), readings=readings)
    for rnd in range(1, max_rounds + 1):
        issues = format_issues(summary, limit, set(paragraphs))
        content_issues, failed = verify_against_evidence(summary, readings, paragraphs, ext, llm, limit,
                                                         stage=f"verify:r{rnd}", num_ctx=cfg.get("verify_num_ctx"))
        issues += content_issues
        rec = {"round": rnd, "issues": issues}
        if failed:
            rec["verify_failed_parts"] = [1]
        result.rounds.append(rec)
        log.info("検証 round %d: 指摘 %d 件%s", rnd, len(issues), " (照合失敗)" if failed else "")
        if not issues:
            result.converged = not failed
            break
        try:
            res = llm.chat(render("revise", summary=summary, issues=_format_issue_list(issues),
                                  paragraph_notes=readings.notes_index(), format=fmt), stage=f"revise:r{rnd}")
        except LLMError as e:  # 修正に失敗しても、それまでの要約は有効なので残す
            log.warning("修正 round %d に失敗、直前の要約を採用して検証を終了: %s", rnd, e)
            rec["revise_failed"] = str(e)
            break
        revised = normalize(res.content)
        if len(split_sections(revised)) < len(split_sections(summary)):
            log.warning("修正版で見出しが減ったため採用しません (round %d)", rnd)
            rec["revise_rejected"] = "見出しが減少"
            break
        summary = shorten_if_needed(revised, llm, limit, fmt, stage=f"shorten:r{rnd}")

    if not result.converged:
        remaining = format_issues(summary, limit, set(paragraphs))
        if remaining:
            log.warning("検証ループ終了時点で形式上の問題が残っています: %s", remaining)
    result.markdown = summary
    return result


# ---- 1) 精読 ------------------------------------------------------------------
@dataclass
class Unit:
    """1 回の精読呼び出しで読む範囲。長い節は段落単位で分割する。"""
    parts: list[tuple[Section, list[Paragraph], bool]]  # (節, 対象段落, 続きか)

    def text(self) -> str:
        out = []
        for sec, paras, cont in self.parts:
            out.append(f"## §{sec.id} {sec.title}{' (続き)' if cont else ''}")
            out += [f"[{p.id}] {p.text}" for p in paras]
        return "\n\n".join(out)

    @property
    def chars(self) -> int:
        return sum(len(p.text) for _, paras, _ in self.parts for p in paras)


def reading_units(sections: list[Section], max_chars: int) -> list[Unit]:
    """読むべき節を順にまとめ、1 単位が max_chars を超えないようにする (長い節は段落の途中で分ける)。"""
    units: list[Unit] = []
    cur = Unit(parts=[])
    for sec in (s for s in sections if s.readable):
        chunk: list[Paragraph] = []
        cont = False
        for p in sec.paragraphs:
            chunk_chars = sum(len(q.text) for q in chunk)
            if (cur.parts or chunk) and cur.chars + chunk_chars + len(p.text) > max_chars:
                if chunk:
                    cur.parts.append((sec, chunk, cont))
                    chunk, cont = [], True
                units.append(cur)
                cur = Unit(parts=[])
            chunk.append(p)
        if chunk:
            cur.parts.append((sec, chunk, cont))
    if cur.parts:
        units.append(cur)
    return units


def read_sections(units: list[Unit], ext: Extracted, title: str, llm: OllamaClient) -> Readings:
    readings = Readings()
    for i, unit in enumerate(units, 1):
        previous = readings.section_summaries(ext.sections)[-3000:] or "(論文の冒頭)"
        expected = {p.id: p for _, paras, _ in unit.parts for p in paras}
        try:
            data = llm.chat_json(render("section_read", title=title, outline=ext.outline, previous=previous,
                                        sections=unit.text()), stage=f"notes:{i}/{len(units)}")
        except LLMError as e:
            log.warning("精読 %d/%d に失敗、原文抜粋で代用: %s", i, len(units), e)
            data = {}
        notes = data.get("paragraphs") or {}
        for pid, p in expected.items():
            note = notes.get(pid)
            readings.notes[pid] = str(note).strip() if note else f"(原文抜粋) {p.text[:200]}"
        missing = [pid for pid in expected if not notes.get(pid)]
        if missing:
            log.warning("精読 %d/%d: 段落メモ欠落 %d/%d 件", i, len(units), len(missing), len(expected))
        by_id = {str(s.get("id", "")).lstrip("§"): s for s in data.get("sections") or [] if isinstance(s, dict)}
        for sec, _paras, cont in unit.parts:
            got = by_id.get(sec.id, {})
            prev = readings.sections.get(sec.id)
            summary = str(got.get("summary", "")).strip()
            if prev and cont:  # 分割された節は要約を連結する
                prev["summary"] = f"{prev['summary']} {summary}".strip()
            else:
                readings.sections[sec.id] = {"title": sec.title, "role": got.get("role", "その他"),
                                             "summary": summary}
    return readings


# ---- 3) 照合 ------------------------------------------------------------------
def cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in CITE_RE.finditer(text):
        for pid in CITE_ID_RE.findall(m.group(1)):
            if pid not in ids:
                ids.append(pid)
    return ids


def verify_against_evidence(summary: str, readings: Readings, paragraphs: dict[str, Paragraph],
                            ext: Extracted, llm: OllamaClient, limit: int, *, stage: str,
                            num_ctx=None, max_evidence_chars: int = 80000) -> tuple[list[dict], bool]:
    """要約が引用した段落の原文だけを渡して照合する。戻り値: (指摘, 照合に失敗したか)。"""
    evidence, total = [], 0
    for pid in cited_ids(summary):
        p = paragraphs.get(pid)
        if p is None:
            continue
        block = f"[{pid}] {p.text[:2500]}"
        if total + len(block) > max_evidence_chars:
            log.warning("根拠段落が多いため一部のみ照合に渡します (%d 文字で打ち切り)", total)
            break
        evidence.append(block)
        total += len(block)
    try:
        data = llm.chat_json(render("verify", summary=summary, evidence="\n\n".join(evidence) or "(なし)",
                                    section_summaries=readings.section_summaries(ext.sections),
                                    section_char_limit=limit),
                             stage=f"{stage}:1/1", num_ctx=int(num_ctx) if num_ctx else None)
    except LLMError as e:
        log.warning("照合に失敗 (%s): %s", stage, e)
        return [], True
    issues = [it for it in data.get("issues", []) or []
              if isinstance(it, dict) and str(it.get("confidence", "high")).lower() != "low"]
    return issues, False


# ---- 短縮・形式 ----------------------------------------------------------------
def shorten_if_needed(summary: str, llm: OllamaClient, limit: int, fmt: str, *, stage: str) -> str:
    """文字数超過の項目だけを推論なしの専用工程で短くする (照合を内容の確認に集中させるため)。"""
    sections = split_sections(summary)
    over = {n: body_chars(b) for n, b in sections.items() if n != FLOW and body_chars(b) > limit}
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
    """前置きやコードフェンスを除去し、最初の見出し (論旨の流れ または 1.) から始まるようにする。"""
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text.strip())
    m = re.search(rf"(?m)^##\s*({FLOW_TITLE}|1\.)", text)
    return (text[m.start():] if m else text).strip() + "\n"


def split_sections(summary: str) -> dict:
    """{"flow": 論旨の流れ, 1: …, …, 6: …}"""
    sections: dict = {}
    for m in re.finditer(rf"(?ms)^##\s*(?:(\d)\.[^\n]*|{FLOW_TITLE}[^\n]*)\n(.*?)(?=^##\s|\Z)", summary):
        sections[int(m.group(1)) if m.group(1) else FLOW] = m.group(2).strip()
    return sections


def strip_citations(text: str) -> str:
    return CITE_RE.sub("", text)


def body_chars(body: str) -> int:
    """箇条書き記号・インデント・改行・根拠表示を除いた文字数。"""
    lines = [re.sub(r"^\s*[-*+]\s*", "", ln).strip() for ln in strip_citations(body).splitlines()]
    return sum(len(ln) for ln in lines)


def _bullets(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if re.match(r"^\s*[-*+]\s", ln)]


def format_issues(summary: str, limit: int, known_ids: set[str] | None = None) -> list[dict]:
    issues: list[dict] = []
    sections = split_sections(summary)

    flow = sections.get(FLOW)
    if flow is None:
        issues.append({"type": "形式", "section": "流れ", "fix": f"見出し「## {FLOW_TITLE}」が欠けている。追加すること"})
    elif not 3 <= len(_bullets(flow)) <= 5:
        issues.append({"type": "形式", "section": "流れ",
                       "fix": "論旨の流れは 3〜5 行 (問題・着想・手法・結果・意義) の箇条書きにすること"})

    for n, title in enumerate(SECTION_TITLES, 1):
        if n not in sections:
            issues.append({"type": "形式", "section": n, "fix": f"見出し「## {title}」が欠けている。追加すること"})
            continue
        body = sections[n]
        chars = body_chars(body)
        if chars > limit:
            issues.append({"type": "形式", "section": n,
                           "fix": f"本文が {chars} 文字で上限 {limit} 文字を超過。重要度の低い記述を削って {limit} 文字以内にすること"})
        if not _bullets(body):
            issues.append({"type": "形式", "section": n, "fix": "階層付き箇条書きになっていない"})

    # 根拠表示: すべての箇条に付いているか、存在する段落 ID か
    for key, body in sections.items():
        no_cite = [b for b in _bullets(body) if not CITE_RE.search(b)]
        if no_cite:
            issues.append({"type": "根拠不備", "section": "流れ" if key == FLOW else key,
                           "summary_text": no_cite[0][:80],
                           "fix": f"根拠表示のない箇条が {len(no_cite)} 件ある。各箇条の文末に段落 ID を付けること"})
        if known_ids is not None:
            unknown = [pid for pid in cited_ids(body) if pid not in known_ids]
            if unknown:
                issues.append({"type": "根拠不備", "section": "流れ" if key == FLOW else key,
                               "fix": f"存在しない段落 ID {unknown[:5]} が使われている。段落ごとの要点から正しい ID を選ぶこと"})
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
