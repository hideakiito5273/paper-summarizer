from paper_summarizer.extract import Extracted
from paper_summarizer.llm import LLMError, LLMResult, parse_json
from paper_summarizer.summarize import summarize

GOOD = "\n".join(f"## {t}\n- x" for t in [
    "1. どんなもの？", "2. 先行研究を比べてどこがすごい？", "3. 技術や手法の肝はどこ？",
    "4. どうやって有効だと検証した？", "5. 議論はある？", "6. 次に読むべき論文は？"])


class FakeLLM:
    parallel = 1

    def __init__(self, verify, merged=GOOD):
        self.verify = verify  # 照合の応答 (dict) または例外
        self.merged = merged  # 統合段階の出力
        self.stages = []

    def chat(self, prompt, *, stage, **kw):
        self.stages.append(stage)
        return LLMResult(self.merged if stage == "merge" else GOOD, "", 1, 1, 0.1, "stop")

    def chat_json(self, prompt, *, stage, **kw):
        self.stages.append(stage)
        if isinstance(self.verify, Exception):
            raise self.verify
        return self.verify


def _ext():
    return Extracted(markdown="## Abstract\ntext\n## Intro\nbody " * 10)


def test_verify_failure_is_not_converged():
    llm = FakeLLM(LLMError("JSON 応答を得られませんでした"))
    res = summarize(_ext(), "T", llm, {"verify_rounds": 3})
    assert not res.converged
    assert res.rounds[0]["verify_failed_parts"] == [1]
    assert len(res.rounds) == 1  # 失敗したまま照合を繰り返さない


def test_no_issues_converges():
    res = summarize(_ext(), "T", FakeLLM({"issues": []}), {"verify_rounds": 3})
    assert res.converged and len(res.rounds) == 1


def test_low_confidence_ignored():
    llm = FakeLLM({"issues": [{"type": "誤り", "section": 1, "confidence": "low"}]})
    assert summarize(_ext(), "T", llm, {"verify_rounds": 3}).converged


def test_parse_json_latex_backslashes():
    assert parse_json(r'{"issues": [{"fix": "\frac{a}{b} と \sigma"}]}')["issues"][0]["fix"] == r"\frac{a}{b} と \sigma"
    assert parse_json('前置き {"issues": []} 後書き') == {"issues": []}


def test_parse_json_valid_escape_latex_not_corrupted():
    # \f は JSON では改ページだが、LaTeX の \frac として保持されること
    assert parse_json(r'{"fix": "\frac{1}{2}"}')["fix"] == r"\frac{1}{2}"
    assert parse_json('{"fix": "a \\"q\\" b"}')["fix"] == 'a "q" b'


def test_shorten_runs_only_when_over_limit():
    llm = FakeLLM({"issues": []})
    summarize(_ext(), "T", llm, {"verify_rounds": 3})
    assert not any(st.startswith("shorten") for st in llm.stages)

    long = GOOD.replace("- x", "- " + "あ" * 600, 1)
    llm = FakeLLM({"issues": []}, merged=long)
    res = summarize(_ext(), "T", llm, {"verify_rounds": 3})
    assert "shorten:merge" in llm.stages
    assert res.converged  # 短縮後は形式の指摘が出ない


def test_run_parallel_keeps_order():
    import time
    from paper_summarizer.summarize_util import run_parallel

    def task(i):
        time.sleep(0.01 * (5 - i))
        return i

    assert run_parallel([lambda i=i: task(i) for i in range(5)], 3) == list(range(5))
    assert run_parallel([lambda i=i: task(i) for i in range(5)], 1) == list(range(5))
