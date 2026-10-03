"""prompts/*.md のテンプレート読み込みと {{var}} 置換。"""

from __future__ import annotations

import hashlib
import re

from .config import PROMPT_DIR


def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **kw) -> str:
    text = load(name)

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in kw:
            raise KeyError(f"prompts/{name}.md の変数 {{{{{key}}}}} が未指定")
        return str(kw[key])

    return re.sub(r"\{\{(\w+)\}\}", sub, text)


def version() -> str:
    """全プロンプトの内容ハッシュ。要約がどのプロンプトで作られたかを meta.json に残す。"""
    h = hashlib.sha256()
    for p in sorted(PROMPT_DIR.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
