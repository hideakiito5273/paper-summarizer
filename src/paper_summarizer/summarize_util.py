"""段階間で共有する小さな実行ユーティリティ。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def run_parallel(tasks: list, parallel: int) -> list:
    """順序を保って実行する。parallel > 1 ならスレッドで同時に投げる (Ollama 側で並列処理される)。
    いずれかのタスクが例外を出した場合はそれを送出する。"""
    if parallel <= 1 or len(tasks) <= 1:
        return [t() for t in tasks]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        return list(ex.map(lambda t: t(), tasks))
