from paper_summarizer.llm import LLMError, LLMResult, OllamaClient


def _res(content):
    return LLMResult(content=content, thinking="t", prompt_tokens=10, eval_tokens=5, duration_s=0.1,
                     done_reason="stop")


def _client(tmp_path, outputs):
    c = OllamaClient({"url": "http://x", "model": "m", "request_retries": 2})
    calls = []

    def fake_stream(body):
        calls.append(body)
        out = outputs[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return _res(out)

    c._stream = fake_stream
    return c, calls


def test_empty_response_is_retried(tmp_path):
    c, calls = _client(tmp_path, ["", "ok"])
    assert c.chat("p", stage="s").content == "ok"
    assert len(calls) == 2


def test_generation_error_is_retried(tmp_path):
    c, calls = _client(tmp_path, [LLMError("token repeat limit reached"), "ok"])
    assert c.chat("p", stage="s").content == "ok"


def test_gives_up_after_retries(tmp_path):
    c, _ = _client(tmp_path, ["", "", ""])
    try:
        c.chat("p", stage="s")
    except LLMError as e:
        assert "空の応答" in str(e)
    else:
        raise AssertionError


def test_cache_reuses_success(tmp_path):
    c, calls = _client(tmp_path, ["first", "second"])
    c.cache_dir = tmp_path / "cache"
    assert c.chat("p", stage="s").content == "first"
    assert c.chat("p", stage="s").content == "first"  # キャッシュから
    assert c.chat("other", stage="s").content == "second"
    assert len(calls) == 2
