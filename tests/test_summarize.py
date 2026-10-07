from paper_summarizer.extract import Extracted
from paper_summarizer.llm import LLMError, LLMResult, parse_json
from paper_summarizer.summarize import summarize

GOOD = """## 論旨の流れ
- 問題: 問題がある。 [§1-p1]
- 手法: 手法を使う。 [§1-p1]
- 結果: 結果が出た。 [§1-p1]
""" + "\n".join(f"## {t}\n- x である。 [§1-p1]" for t in [
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
        if stage.startswith("notes"):
            return {"sections": [{"id": "1", "role": "手法", "summary": "要約。"}], "paragraphs": {"§1-p1": "要点。"}}
        if isinstance(self.verify, Exception):
            raise self.verify
        return self.verify


def _ext():
    from paper_summarizer.structure import Paragraph, Section
    sec = Section(id="1", title="1. Intro", number="1", paragraphs=[Paragraph("§1-p1", "text", "本文。")])
    return Extracted(markdown="## 1. Intro\n本文。", sections=[sec])


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

    long = GOOD.replace("## 1. どんなもの？\n- x", "## 1. どんなもの？\n- " + "あ" * 1100, 1)
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


def test_parse_json_odd_backslash_runs():
    d = parse_json(r'{"a": "D_t \\\ D_{t-1} と O\'Hagan と \"q\""}')
    assert d["a"] == "D_t \\\\ D_{t-1} と O\\'Hagan と \"q\""


def test_parse_reading_text_format():
    from paper_summarizer.summarize import parse_reading
    out = parse_reading("""[節 §5]
役割: 理論
要約: 累積粗さモデルを提案する。
事後分布を導出する。
[節 §6]
役割: 考察
要約: 適用範囲を論じる。
[段落]
§5-p1: D_t \\ D_{t-1} を用いる。
§5-p2: O'Hagan (1996) の手法で "E(L)" を求める。
  続きの行。
- §6-p1: 広く適用できる。
""")
    assert [s["id"] for s in out["sections"]] == ["5", "6"]
    assert out["sections"][0]["summary"] == "累積粗さモデルを提案する。 事後分布を導出する。"
    assert out["paragraphs"]["§5-p2"].endswith("続きの行。")
    assert set(out["paragraphs"]) == {"§5-p1", "§5-p2", "§6-p1"}


def test_parse_json_missing_closing_brace():
    assert parse_json('{"issues": [{"fix": "a"}]')["issues"][0]["fix"] == "a"


def test_changed_bullets():
    from paper_summarizer.summarize import changed_bullets
    new = GOOD.replace("- x である。 [§1-p1]", "- y である。 [§1-p1]", 1)
    assert changed_bullets(GOOD, new) == ["- y である。 [§1-p1]"]
    assert changed_bullets(GOOD, GOOD.replace("。 [", "。  [")) == []  # 空白の違いは無視


class IncrementalLLM(FakeLLM):
    """1 回目の照合で指摘 → 修正で 1 箇条だけ変える → 2 回目は変わった箇条だけ照合されるか。"""

    def __init__(self, verdicts):
        super().__init__({"issues": []})
        self.verdicts = list(verdicts)
        self.verify_prompts = []
        self.n_revise = 0

    def chat(self, prompt, *, stage, **kw):
        self.stages.append(stage)
        if stage.startswith("revise"):
            self.n_revise += 1
            return LLMResult(GOOD.replace("- x である。 [§1-p1]", f"- 修正{self.n_revise}。 [§1-p1]", 1),
                             "", 1, 1, 0.1, "stop")
        return LLMResult(GOOD, "", 1, 1, 0.1, "stop")

    def chat_json(self, prompt, *, stage, **kw):
        if stage.startswith("verify"):
            self.stages.append(stage)
            self.verify_prompts.append(prompt)
            return self.verdicts.pop(0)
        return super().chat_json(prompt, stage=stage, **kw)


def test_incremental_verification_scope():
    issue = {"issues": [{"type": "誤り", "section": "流れ", "fix": "直す"}]}
    llm = IncrementalLLM([issue, {"issues": []}])
    res = summarize(_ext(), "T", llm, {"verify_rounds": 3})
    assert res.converged and [r["scope"] for r in res.rounds] == ["all", 1]
    assert "要約のすべての箇条" in llm.verify_prompts[0] and "欠落:" in llm.verify_prompts[0]
    assert "修正1。" in llm.verify_prompts[1] and "欠落の確認は不要" in llm.verify_prompts[1]


def test_final_revision_is_verified_then_stops():
    issue = {"issues": [{"type": "誤り", "section": 1, "fix": "直す"}]}
    llm = IncrementalLLM([issue] * 5)
    res = summarize(_ext(), "T", llm, {"verify_rounds": 2})
    # 修正 2 回 + 最後の修正後の照合 = 照合 3 回、指摘が残ったので未収束
    assert len(res.rounds) == 3 and llm.n_revise == 2 and not res.converged
