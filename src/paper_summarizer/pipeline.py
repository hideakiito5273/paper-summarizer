"""inbox の走査 → 未処理 PDF の要約 → library への出力。"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import prompts
from .config import Config
from .db import DB, now
from .extract import Extractor
from .llm import OllamaClient, OllamaUnavailable
from .output import archive_previous, paper_dir_name, unique_dir, update_project_readme, write_outputs
from .summarize import summarize

log = logging.getLogger(__name__)

UNSORTED = "_unsorted"


@dataclass
class RunReport:
    done: list[dict] = field(default_factory=list)      # {"title", "project", "id"}
    failed: list[dict] = field(default_factory=list)    # {"name", "project", "error", "final"}
    duplicates: list[dict] = field(default_factory=list)
    aborted: str | None = None

    @property
    def has_news(self) -> bool:
        return bool(self.done or self.failed or self.duplicates or self.aborted)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_candidate(path: Path, inbox: Path) -> bool:
    rel = path.relative_to(inbox)
    if any(part.startswith((".", "~")) for part in rel.parts):  # .stversions, .syncthing.*.tmp など
        return False
    return path.is_file() and path.suffix.lower() == ".pdf"


def project_of(path: Path, inbox: Path) -> str:
    rel = path.relative_to(inbox)
    return rel.parts[0] if len(rel.parts) > 1 else UNSORTED


def move_to_failed(cfg: Config, pdf: Path, project: str, reason: str) -> Path:
    dest_dir = cfg.paths.failed / project
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf.name
    if dest.exists():
        dest = dest_dir / f"{pdf.stem}_{datetime.now():%Y%m%d-%H%M%S}{pdf.suffix}"
    if pdf.exists():
        shutil.move(str(pdf), dest)
    dest.with_name(dest.name + ".error.txt").write_text(f"{now()}\n{reason}\n", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
def scan(cfg: Config, db: DB, report: RunReport) -> None:
    inbox = cfg.paths.inbox
    min_age = float(cfg.scan.get("min_age_seconds", 120))
    t_now = time.time()

    for pdf in sorted(inbox.rglob("*")):
        if not _is_candidate(pdf, inbox):
            continue
        if t_now - pdf.stat().st_mtime < min_age:
            log.info("更新直後のため次回に回します: %s", pdf)
            continue
        project = project_of(pdf, inbox)
        sha = sha256_file(pdf)

        existing = db.find_by_inbox_path(str(pdf))
        if existing:
            if existing["sha256"] != sha:  # 処理待ちの間に中身が差し替わった
                log.info("処理待ちファイルの内容が変わりました: %s", pdf)
                db.update(existing["id"], sha256=sha, attempts=0, last_error=None)
            continue

        queued = db.find_by_sha(sha, ("pending", "processing"))
        if queued:
            dest = move_to_failed(cfg, pdf, project, f"重複: 処理待ちの {queued['inbox_path']} と同一内容")
            db.add(sha256=sha, project=project, source_name=pdf.name, inbox_path=str(dest),
                   status="duplicate", last_error=f"duplicate of #{queued['id']}")
            report.duplicates.append({"name": pdf.name, "project": project, "of": queued["inbox_path"]})
            continue

        dup = db.find_by_sha(sha, ("done",))
        if dup and dup["project"] == project:
            dest = move_to_failed(cfg, pdf, project, f"重複: 処理済みの {dup['output_dir']} と同一内容")
            db.add(sha256=sha, project=project, source_name=pdf.name, inbox_path=str(dest),
                   status="duplicate", last_error=f"duplicate of #{dup['id']}")
            report.duplicates.append({"name": pdf.name, "project": project, "of": dup["output_dir"]})
            log.info("重複のため failed/ へ移動: %s (#%d と同一)", pdf.name, dup["id"])
            continue
        if dup:  # 他プロジェクトで処理済み → 成果物を複製して再要約を省く
            _copy_from_other_project(cfg, db, dup, pdf, project, sha, report)
            continue

        replaces = db.find_done_by_name(project, pdf.name)
        pid = db.add(sha256=sha, project=project, source_name=pdf.name, inbox_path=str(pdf),
                     replaces=replaces["id"] if replaces else None)
        log.info("登録 #%d: %s/%s%s", pid, project, pdf.name,
                 f" (#{replaces['id']} の差し替え)" if replaces else "")

    for row in db.pending():
        if not Path(row["inbox_path"]).exists():
            db.update(row["id"], status="missing", last_error="処理前に inbox から削除された")
            log.warning("inbox から消えました: #%d %s", row["id"], row["inbox_path"])


def _copy_from_other_project(cfg: Config, db: DB, dup, pdf: Path, project: str, sha: str,
                             report: RunReport) -> None:
    src = Path(dup["output_dir"])
    dest = unique_dir(cfg.paths.library / project, src.name)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("_history"))
    pdf.unlink()
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(project=project, copied_from=str(src))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pid = db.add(sha256=sha, project=project, source_name=pdf.name, inbox_path=str(pdf), status="done")
    db.update(pid, title=dup["title"], authors=dup["authors"], year=dup["year"], venue=dup["venue"],
              doi=dup["doi"], output_dir=str(dest), model=dup["model"], prompt_version=dup["prompt_version"],
              finished_at=now())
    update_project_readme(db, cfg.paths.reviews, cfg.paths.library, project)
    report.done.append({"id": pid, "title": dup["title"], "project": project, "note": "他プロジェクトから複製"})
    log.info("他プロジェクトの要約を複製 #%d → %s", dup["id"], dest)


# ---------------------------------------------------------------------------
def process_pending(cfg: Config, db: DB, llm: OllamaClient, report: RunReport) -> None:
    rows = db.pending()
    if not rows:
        log.info("未処理の論文はありません")
        return
    llm.check([cfg.ollama["model"], cfg.ollama.get("vision_model", cfg.ollama["model"])])
    extractor = Extractor(cfg.extract)
    max_attempts = int(cfg.scan.get("max_attempts", 3))

    for row in rows:
        pid = row["id"]
        pdf = Path(row["inbox_path"])
        attempts = row["attempts"] + 1
        db.update(pid, status="processing", attempts=attempts, started_at=now())
        llm.paper_id = pid
        log.info("==== 処理開始 #%d %s/%s (試行 %d/%d)", pid, row["project"], row["source_name"],
                 attempts, max_attempts)
        t0 = time.monotonic()
        try:
            meta = _process_one(cfg, db, llm, extractor, row, pdf)
        except OllamaUnavailable as e:
            db.update(pid, status="pending", attempts=attempts - 1, last_error=str(e))
            report.aborted = str(e)
            log.error("Ollama が利用できないため中断: %s", e)
            return
        except Exception as e:  # noqa: BLE001 — 1 本の失敗で全体を止めない
            err = f"{type(e).__name__}: {e}"
            log.error("処理失敗 #%d: %s\n%s", pid, err, traceback.format_exc())
            if attempts >= max_attempts:
                dest = move_to_failed(cfg, pdf, row["project"],
                                      f"{max_attempts} 回失敗\n{err}\n\n{traceback.format_exc()}")
                db.update(pid, status="failed", last_error=err, inbox_path=str(dest), finished_at=now())
                report.failed.append({"name": row["source_name"], "project": row["project"],
                                      "error": err, "final": True})
            else:
                db.update(pid, status="pending", last_error=err)
                report.failed.append({"name": row["source_name"], "project": row["project"],
                                      "error": err, "final": False})
            continue
        finally:
            llm.paper_id = None
            llm.cache_dir = None

        dur = time.monotonic() - t0
        db.update(pid, status="done", finished_at=now(), duration_s=round(dur, 1), last_error=None)
        if row["replaces"]:
            db.update(row["replaces"], status="superseded")
        update_project_readme(db, cfg.paths.reviews, cfg.paths.library, row["project"])
        report.done.append({"id": pid, "title": meta.get("title") or row["source_name"],
                            "project": row["project"], "minutes": round(dur / 60, 1)})
        log.info("==== 処理完了 #%d (%.1f 分)", pid, dur / 60)


def first_page_text(pdf: Path, max_chars: int = 3000) -> str:
    try:
        import pypdfium2

        doc = pypdfium2.PdfDocument(str(pdf))
        try:
            return doc[0].get_textpage().get_text_range()[:max_chars]
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001 — 取れなくても抽出本文だけで続行する
        log.warning("1 ページ目の生テキストを取得できません: %r", e)
        return "(取得不可)"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _process_one(cfg: Config, db: DB, llm: OllamaClient, extractor: Extractor, row, pdf: Path) -> dict:
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    # 作業ディレクトリは成功時のみ削除する。再試行時は LLM 応答キャッシュで成功済みの段階を再利用する
    work = cfg.paths.work_dir / str(row["id"])
    if row["sha256"] != _read_text(work / "sha256"):
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / "sha256").write_text(row["sha256"])
    llm.cache_dir = work / "llm_cache"
    vision_model = cfg.ollama.get("vision_model", cfg.ollama["model"])

    ext = extractor.extract(pdf, work, llm, vision_model)
    if len(ext.markdown.strip()) < 500:
        raise ValueError("本文をほとんど抽出できませんでした (スキャン PDF の場合は extract.ocr = true)")

    # Docling は欄外 (例: "Biometrika (2000), 87, 1") を除去するため、PDF の生テキストも渡す
    head = f"## PDF 1 ページ目の生テキスト (欄外を含む)\n{first_page_text(pdf)}\n\n## 抽出本文の冒頭\n{ext.head}"
    meta = llm.chat_json(prompts.render("metadata", head=head), stage="metadata")
    meta = {k: meta.get(k) for k in ("title", "authors", "year", "venue", "doi", "short_title")}
    if not isinstance(meta.get("authors"), list):
        meta["authors"] = [meta["authors"]] if meta.get("authors") else []
    try:
        meta["year"] = int(meta["year"]) if meta.get("year") else None
    except (TypeError, ValueError):
        meta["year"] = None

    result = summarize(ext, meta.get("title") or row["source_name"], llm, cfg.summarize)

    if row["replaces"]:
        old = db.get(row["replaces"])
        out_dir = Path(old["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_previous(out_dir)
    else:
        out_dir = unique_dir(cfg.paths.library / row["project"], paper_dir_name(meta))

    meta.update(
        project=row["project"],
        source_name=row["source_name"],
        sha256=row["sha256"],
        processed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        model=cfg.ollama["model"],
        vision_model=vision_model,
        prompt_version=prompts.version(),
        n_chunks=result.n_chunks,
        n_figures=len(ext.figures),
        verify_rounds=len(result.rounds),
        verify_converged=result.converged,
    )
    verification = {"converged": result.converged, "rounds": result.rounds}
    write_outputs(out_dir, pdf, meta, result.markdown, verification, work / "paper.md")
    if (work / "figures").exists():
        shutil.copytree(work / "figures", out_dir / "figures", dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)

    db.update(row["id"], title=meta.get("title"), authors=meta.get("authors"), year=meta.get("year"),
              venue=meta.get("venue"), doi=meta.get("doi"), output_dir=str(out_dir),
              model=cfg.ollama["model"], prompt_version=meta["prompt_version"], inbox_path=None)
    return meta
