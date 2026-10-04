"""Ollama /api/chat クライアント。

- ストリーミングで受信し、トークンが idle_timeout 秒途切れたらタイムアウトとする
  (全体の処理時間には上限を設けない)。
- Ollama は num_ctx を超えた入力を黙って切り詰めるため、prompt_eval_count で検知して例外にする。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ContextOverflow(LLMError):
    pass


class OllamaUnavailable(LLMError):
    pass


@dataclass
class LLMResult:
    content: str
    thinking: str
    prompt_tokens: int | None
    eval_tokens: int | None
    duration_s: float
    done_reason: str | None


class OllamaClient:
    def __init__(self, cfg: dict, db=None, run_id: str | None = None):
        self.url = cfg["url"].rstrip("/")
        self.cfg = cfg
        self.db = db
        self.run_id = run_id
        self.paper_id: int | None = None  # ログ記録用に呼び出し側が設定する

    # ------------------------------------------------------------------
    def check(self, models: list[str]) -> None:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=10)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaUnavailable(f"Ollama に接続できません ({self.url}): {e}") from e
        available = {m["name"] for m in r.json().get("models", [])}
        missing = [m for m in set(models) if m not in available and f"{m}:latest" not in available]
        if missing:
            raise OllamaUnavailable(f"モデルが見つかりません: {missing} (ollama pull が必要)")

    # ------------------------------------------------------------------
    def chat(self, prompt: str, *, stage: str, model: str | None = None, images: list[Path] | None = None,
             system: str | None = None, think: bool | str | None = None,
             num_ctx: int | None = None) -> LLMResult:
        model = model or self.cfg["model"]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        msg: dict = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        messages.append(msg)

        num_ctx = int(num_ctx or self.cfg.get("num_ctx", 32768))
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.cfg.get("keep_alive", "30m"),
            "think": self.cfg.get("think", True) if think is None else think,
            "options": {
                "num_ctx": num_ctx,
                "temperature": float(self.cfg.get("temperature", 0.3)),
                "num_predict": int(self.cfg.get("num_predict", 32768)),
            },
        }

        retries = int(self.cfg.get("request_retries", 2))
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            t0 = time.monotonic()
            try:
                res = self._stream(body)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_err = e
                dur = time.monotonic() - t0
                self._record(stage, model, None, None, dur, ok=False)
                log.warning("LLM 呼び出し失敗 stage=%s attempt=%d/%d: %r", stage, attempt + 1, retries + 1, e)
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            except LLMError as e:  # 生成中の Ollama エラー (繰り返しループによる打ち切り等) はサンプリングし直す
                last_err = e
                self._record(stage, model, None, None, time.monotonic() - t0, ok=False)
                log.warning("LLM 生成エラー stage=%s attempt=%d/%d: %s", stage, attempt + 1, retries + 1, e)
                continue
            self._record(stage, model, res.prompt_tokens, res.eval_tokens, res.duration_s, ok=True)
            log.info(
                "LLM stage=%s model=%s prompt_tok=%s eval_tok=%s %.1fs done=%s",
                stage, model, res.prompt_tokens, res.eval_tokens, res.duration_s, res.done_reason,
            )
            if res.prompt_tokens and res.prompt_tokens >= num_ctx - 16:
                raise ContextOverflow(
                    f"入力 ({res.prompt_tokens} tok) が num_ctx={num_ctx} を超え切り詰められた可能性 (stage={stage})"
                )
            if res.done_reason == "length":
                log.warning("出力が num_predict 上限で打ち切られました stage=%s", stage)
            if not res.content.strip():
                raise LLMError(f"空の応答 (stage={stage})")
            return res
        raise LLMError(f"LLM 呼び出しが {retries + 1} 回失敗 (stage={stage}): {last_err!r}")

    def chat_json(self, prompt: str, *, stage: str, **kw) -> dict:
        """JSON 応答を得る。Ollama の format=json は生成が大幅に遅くなるため使わず、
        プロンプトで指示した出力をパースする。失敗時は 1 回だけ再生成する。"""
        for attempt in range(2):
            res = self.chat(prompt, stage=stage, **kw)
            try:
                return parse_json(res.content)
            except (json.JSONDecodeError, ValueError):
                log.warning("JSON をパースできませんでした stage=%s (attempt %d)", stage, attempt + 1)
        raise LLMError(f"JSON 応答を得られませんでした (stage={stage})")

    # ------------------------------------------------------------------
    def _stream(self, body: dict) -> LLMResult:
        timeout = httpx.Timeout(
            connect=float(self.cfg.get("connect_timeout_seconds", 30)),
            read=float(self.cfg.get("idle_timeout_seconds", 900)),
            write=60.0,
            pool=60.0,
        )
        content: list[str] = []
        thinking: list[str] = []
        final: dict = {}
        t0 = time.monotonic()
        with httpx.stream("POST", f"{self.url}/api/chat", json=body, timeout=timeout) as r:
            if r.status_code >= 400:
                r.read()
                raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:500]}", request=r.request, response=r)
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(f"Ollama エラー: {chunk['error']}")
                m = chunk.get("message", {})
                if m.get("content"):
                    content.append(m["content"])
                if m.get("thinking"):
                    thinking.append(m["thinking"])
                if chunk.get("done"):
                    final = chunk
        return LLMResult(
            content=strip_think("".join(content)),
            thinking="".join(thinking),
            prompt_tokens=final.get("prompt_eval_count"),
            eval_tokens=final.get("eval_count"),
            duration_s=time.monotonic() - t0,
            done_reason=final.get("done_reason"),
        )

    def _record(self, stage, model, prompt_tokens, eval_tokens, dur, ok) -> None:
        if self.db is not None:
            self.db.log_llm_call(run_id=self.run_id, paper_id=self.paper_id, stage=stage, model=model,
                                 prompt_tokens=prompt_tokens, eval_tokens=eval_tokens, duration_s=dur, ok=ok)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """think 指定が効かないモデルで本文に <think> が混入した場合の除去。"""
    return _THINK_RE.sub("", text).strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
