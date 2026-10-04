"""Docling による PDF → Markdown 抽出と、図・数式画像の VLM 読み取り。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError, OllamaClient, OllamaUnavailable
from .prompts import render

log = logging.getLogger(__name__)

FIG_MARK = "<!-- image -->"
MIN_FIGURE_PX = 120  # これより小さい画像 (ロゴ等) は読み取らない


@dataclass
class Figure:
    index: int
    path: Path
    caption: str
    description: str = ""


@dataclass
class Extracted:
    markdown: str
    figures: list[Figure] = field(default_factory=list)
    n_formulas_vlm: int = 0

    @property
    def outline(self) -> str:
        return "\n".join(line for line in self.markdown.splitlines() if line.startswith("#"))

    @property
    def abstract(self) -> str:
        m = re.search(r"^#+\s*abstract\s*$(.*?)(?=^#)", self.markdown, re.I | re.M | re.S)
        text = m.group(1) if m else self.markdown[:3000]
        return text.strip()[:4000]

    @property
    def head(self) -> str:
        return self.markdown[:6000]


class Extractor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions()
            opts.do_ocr = bool(self.cfg.get("ocr", False))
            opts.do_table_structure = True
            opts.table_structure_options.mode = (
                TableFormerMode.ACCURATE if self.cfg.get("table_mode", "accurate") == "accurate"
                else TableFormerMode.FAST
            )
            opts.table_structure_options.do_cell_matching = True
            opts.do_formula_enrichment = bool(self.cfg.get("formula_enrichment", True))
            opts.generate_page_images = True
            opts.generate_picture_images = True
            opts.images_scale = float(self.cfg.get("image_scale", 2.0))
            opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
            if self.cfg.get("artifacts_path"):
                opts.artifacts_path = Path(self.cfg["artifacts_path"]).expanduser()
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
            log.info("Docling converter 初期化完了")
        return self._converter

    def extract(self, pdf: Path, work_dir: Path, llm: OllamaClient | None, vision_model: str | None) -> Extracted:
        from docling_core.types.doc import FormulaItem, PictureItem

        work_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = work_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        log.info("Docling 抽出開始: %s", pdf.name)
        doc = self._get_converter().convert(pdf).document

        pictures: list = []
        formulas: list = []
        for item, _level in doc.iterate_items():  # Markdown 出力と同じ読み順
            if isinstance(item, PictureItem):
                pictures.append(item)
            elif isinstance(item, FormulaItem):
                formulas.append(item)

        # 数式: "all" は全数式を VLM で読み直す (Docling は誤読しても空にならないため)。
        #       "fallback" は Docling がデコードできなかったものだけ。
        mode = self.cfg.get("formula_vlm", "all")
        n_vlm = 0
        for j, item in enumerate(formulas):
            if llm is None or mode == "off" or (mode == "fallback" and (item.text or "").strip()):
                continue
            img = item.get_image(doc)
            if img is None:
                continue
            p = fig_dir / f"formula_{j:03d}.png"
            img.save(p)
            try:
                res = llm.chat(render("formula", docling=item.text or "(なし)"), stage="formula",
                               model=vision_model, images=[p], think=False)
            except OllamaUnavailable:
                raise
            except LLMError as e:  # 1 つの数式の失敗で論文全体を落とさない (Docling の結果を使う)
                log.warning("数式 %d の VLM 読み取りに失敗、Docling の結果を使用: %s", j, e)
                continue
            latex = _clean_latex(res.content)
            if latex and latex != "判読不可":
                item.text = latex
                n_vlm += 1

        markdown = doc.export_to_markdown(image_placeholder=FIG_MARK)

        figures: list[Figure] = []
        for i, pic in enumerate(pictures):
            img = pic.get_image(doc)
            path = fig_dir / f"fig_{i:03d}.png"
            caption = pic.caption_text(doc) or ""
            if img is not None:
                img.save(path)
            figures.append(Figure(index=i, path=path if img is not None else Path(), caption=caption))

        describe = self.cfg.get("describe_figures", True) and llm is not None
        max_figs = int(self.cfg.get("max_figures", 40))
        for fig in figures:
            if not describe or fig.index >= max_figs or not fig.path.is_file():
                continue
            if _too_small(fig.path):
                fig.description = "(小さな画像のため省略)"
                continue
            try:
                res = llm.chat(render("figure", caption=fig.caption or "(なし)"), stage="figure",
                               model=vision_model, images=[fig.path], think=False)
            except OllamaUnavailable:
                raise
            except LLMError as e:
                log.warning("図 %d の VLM 読み取りに失敗、キャプションのみ使用: %s", fig.index + 1, e)
                fig.description = "(読み取り失敗)"
                continue
            fig.description = res.content.strip()

        markdown = _replace_in_order(markdown, FIG_MARK, [_figure_block(f) for f in figures])

        (work_dir / "paper.md").write_text(markdown, encoding="utf-8")
        log.info("抽出完了: %d 文字, 図 %d, VLM 数式 %d", len(markdown), len(figures), n_vlm)
        return Extracted(markdown=markdown, figures=figures, n_formulas_vlm=n_vlm)


def _clean_latex(text: str) -> str:
    text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text.strip())
    return text.strip().strip("$").strip()


def _too_small(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as im:
        return min(im.size) < MIN_FIGURE_PX


def _figure_block(f: Figure) -> str:
    cap = f.caption.strip() or "(キャプションなし)"
    desc = f.description.strip() or "(読み取りなし)"
    return f"[図 {f.index + 1}: {cap}]\n[図の内容: {desc}]"


def _replace_in_order(text: str, mark: str, replacements: list[str]) -> str:
    parts = text.split(mark)
    if len(parts) - 1 != len(replacements):
        log.warning("プレースホルダ数 (%d) と置換数 (%d) が一致しません", len(parts) - 1, len(replacements))
    out = [parts[0]]
    for i, part in enumerate(parts[1:]):
        out.append(replacements[i] if i < len(replacements) else mark)
        out.append(part)
    return "".join(out)


def split_chunks(markdown: str, max_chars: int) -> list[str]:
    """見出し単位で分割し、max_chars を超えないようにまとめる。長すぎる節は段落で分割。"""
    sections = re.split(r"(?m)^(?=#{1,6} )", markdown)
    pieces: list[str] = []
    for sec in sections:
        if len(sec) <= max_chars:
            pieces.append(sec)
            continue
        buf = ""
        for para in re.split(r"\n{2,}", sec):
            if buf and len(buf) + len(para) + 2 > max_chars:
                pieces.append(buf)
                buf = ""
            while len(para) > max_chars:  # 段落単体でも長すぎる場合
                pieces.append(para[:max_chars])
                para = para[max_chars:]
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = ""
        cur += p
    if cur.strip():
        chunks.append(cur)
    return chunks
