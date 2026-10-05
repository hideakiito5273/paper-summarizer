"""Docling 文書から「節 → 段落」の構造を組み立てる。

段落には根拠表示用の ID (例: §3.2-p4) を振る。
- 節 ID: 見出しの番号 (「2·1.」→ 2.1) を優先し、なければ Abs / Intro / Ack / App / Ref / Alg1 / S7 など
- 段落: 本文・箇条書き (連続する項目をまとめる)・表・図 (VLM の読み取り結果を含む)。数式は直前の段落に付ける
- 参考文献の節は 1 項目 = 1 段落 とし、読書メモの対象外にする
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Adobe 系フォントの small caps は私用領域 (U+F720〜U+F77E) に ASCII + 0xF700 で符号化されることがある
_PUA_RE = re.compile("[-]")
_SPACED_RE = re.compile(r"(?:\w ){3,}\w")  # "a b s t r a c t" のような字間空け
_NUM_RE = re.compile(r"^\s*((?:\d+|[A-Z])(?:[.·]\d+)*)\.?\s+(.*)$")
_YEAR_RE = re.compile(r"\b(1[89]|20)\d{2}[a-z]?\b")


def fix_text(text: str) -> str:
    text = _PUA_RE.sub(lambda m: chr(ord(m.group()) - 0xF700), text or "")
    if _SPACED_RE.fullmatch(text.strip()):
        text = text.replace(" ", "")
    return text


@dataclass
class Paragraph:
    id: str
    kind: str  # text / list / table / figure / formula / reference
    text: str


@dataclass
class Section:
    id: str
    title: str
    number: str | None = None
    paragraphs: list[Paragraph] = field(default_factory=list)
    kind: str = "body"  # front / abstract / body / ack / references

    @property
    def depth(self) -> int:
        return self.number.count(".") + 1 if self.number else 1

    @property
    def readable(self) -> bool:
        """読書メモの対象か (前付け・謝辞・参考文献は除く)。"""
        return self.kind in ("abstract", "body") and bool(self.paragraphs)

    def text(self, max_chars: int | None = None) -> str:
        body = "\n\n".join(f"[{p.id}] {p.text}" for p in self.paragraphs)
        return body if max_chars is None else body[:max_chars]


def _key(title: str) -> str:
    """判定用: 空白を除いた小文字 ("REFERENCE S" → "references")。"""
    return re.sub(r"\s+", "", title).lower()


def _section_id(title: str, number: str | None, index: int) -> str:
    if number:
        return number
    t = _key(title)
    m = re.match(r"(algorithm|table|figure)(\d+)", t)
    if m:
        return {"algorithm": "Alg", "table": "Tab", "figure": "Fig"}[m.group(1)] + m.group(2)
    for pat, sid in ((r"abstract", "Abs"), (r"summary", "Sum"), (r"introduction", "Intro"),
                     (r"conclu", "Concl"), (r"acknowledg", "Ack"), (r"appendix", "App"),
                     (r"references|bibliography|literaturecited", "Ref"), (r"discussion", "Disc")):
        if re.match(pat, t):
            return sid
    return f"S{index}"


def _kind_of(title: str, number: str | None, seen_body: bool) -> str:
    t = _key(title)
    if re.match(r"references|bibliography|literaturecited", t):
        return "references"
    if re.match(r"acknowledg", t):
        return "ack"
    if not seen_body and not number and re.match(r"abstract|summary", t):
        return "abstract"
    if not seen_body and not number:
        return "front"
    return "body"


def _looks_like_references(sec: Section) -> bool:
    items = [p for p in sec.paragraphs if p.kind in ("list", "text", "reference")]
    if len(items) < 3:
        return False
    hits = sum(1 for p in items if _YEAR_RE.search(p.text) and len(p.text) < 600)
    return hits / len(items) >= 0.6


def build_sections(doc, figure_blocks: dict[int, str] | None = None) -> list[Section]:
    """doc: DoclingDocument。figure_blocks: id(PictureItem) → 本文に差し込む図の記述。"""
    from docling_core.types.doc import (
        FormulaItem, ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem,
    )

    figure_blocks = figure_blocks or {}
    sections: list[Section] = [Section(id="Front", title="(前付け)", kind="front")]
    seen_body = False
    used_ids: set[str] = {"Front"}

    def new_section(title: str) -> Section:
        nonlocal seen_body
        title = fix_text(title).strip()
        m = _NUM_RE.match(title)
        number = m.group(1).replace("·", ".") if m and any(c.isdigit() for c in m.group(1)) else None
        name = m.group(2) if number else title
        kind = _kind_of(name, number, seen_body)
        if kind == "body":
            seen_body = True
        sid = _section_id(name, number, len(sections))
        base, k = sid, 2
        while sid in used_ids:
            sid, k = f"{base}_{k}", k + 1
        used_ids.add(sid)
        return Section(id=sid, title=title, number=number, kind=kind)

    def add(sec: Section, kind: str, text: str) -> None:
        text = fix_text(text).strip()
        if not text:
            return
        if sec.kind == "references":
            kind = "reference"
        sec.paragraphs.append(Paragraph(id=f"§{sec.id}-p{len(sec.paragraphs) + 1}", kind=kind, text=text))

    for item, _level in doc.iterate_items():
        sec = sections[-1]
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            if isinstance(item, TitleItem) and not seen_body:
                sec.paragraphs.append(Paragraph(id=f"§{sec.id}-p{len(sec.paragraphs) + 1}", kind="text",
                                                text=fix_text(item.text)))
                continue
            sections.append(new_section(item.text))
            continue
        label = str(getattr(item, "label", "")).split(".")[-1]
        if label in ("caption", "page_header", "page_footer"):
            continue  # キャプションは図・表の段落に含める
        if isinstance(item, PictureItem):
            block = figure_blocks.get(id(item))
            if block:
                add(sec, "figure", block)
        elif isinstance(item, TableItem):
            cap = item.caption_text(doc) or ""
            try:
                table_md = item.export_to_markdown(doc)
            except Exception:  # noqa: BLE001 — 表の書き出しに失敗してもキャプションは残す
                table_md = ""
            add(sec, "table", f"[表: {cap}]\n{table_md[:3000]}")
        elif isinstance(item, FormulaItem):
            latex = f"$${(item.text or '').strip()}$$" if (item.text or "").strip() else ""
            if latex and sec.paragraphs and sec.paragraphs[-1].kind in ("text", "list"):
                sec.paragraphs[-1].text += f"\n{latex}"  # 数式は直前の段落の一部として扱う
            else:
                add(sec, "formula", latex)
        elif isinstance(item, ListItem):
            text = fix_text(item.text).strip()
            if sec.kind != "references" and sec.paragraphs and sec.paragraphs[-1].kind == "list":
                sec.paragraphs[-1].text += f"\n- {text}"
            else:
                add(sec, "list", text if sec.kind == "references" else f"- {text}")
        elif isinstance(item, TextItem):
            add(sec, "text", item.text)

    # 見出しで判別できなかった参考文献の節を中身で判定する (末尾側のみ)
    for sec in reversed(sections):
        if sec.kind == "body" and _looks_like_references(sec):
            sec.kind = "references"
            for p in sec.paragraphs:
                p.kind = "reference"
            break
        if sec.kind == "body" and sec.paragraphs:
            break
    return [s for s in sections if s.paragraphs or s.kind == "references"]


def paragraph_index(sections: list[Section]) -> dict[str, Paragraph]:
    return {p.id: p for s in sections for p in s.paragraphs}


def outline(sections: list[Section]) -> str:
    return "\n".join(f"{'  ' * (s.depth - 1)}§{s.id} {s.title} ({len(s.paragraphs)} 段落, {s.kind})"
                     for s in sections)
