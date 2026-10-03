# paper-summarizer

DGX Spark 上のローカル LLM (Ollama) で論文 PDF を自動要約するシステム。
論文ファイルは外部サービスに送信しない (共有は Syncthing の P2P 同期、推論はローカル)。

## フォルダ構成

```
/data/papers/
├── inbox/<プロジェクト>/*.pdf   ← ここに置く (直下に置くと _unsorted 扱い)
├── library/<プロジェクト>/<年>_<第一著者>_<短題>/
│   ├── paper.pdf / summary.md / meta.json
│   ├── verification.json        ← 自己検証の指摘履歴
│   ├── extracted.md / figures/  ← 抽出結果 (図の読み取り結果を含む)
│   └── _history/                ← 差し替え前の旧版
├── reviews/<プロジェクト>/README.md, review_YYYY-MM-DD.md
├── failed/<プロジェクト>/*.pdf + *.error.txt
└── .state/                      ← SQLite・作業領域 (Syncthing 同期対象外)
```

## 処理の流れ

1. `inbox/` を走査し、SHA-256 で SQLite に登録 (同一内容は重複、同名は差し替えとして再要約)
2. Docling で PDF → Markdown (表構造・数式 LaTeX)。図と未デコード数式は VLM で読み取り本文へ差し込む
3. 書誌情報抽出 → 見出し単位で分割し読書メモ作成 → 統合して 6 項目要約 (`prompts/paper_summary.md`)
4. 自己検証ループ: 形式チェック (見出し・500 字) + 全チャンクとの照合 → 修正 (最大 `verify_rounds` 回)
5. `library/` へ出力、プロジェクト README 更新、Gmail で件数とタイトルを通知

## コマンド

```bash
.venv/bin/paper-summarizer run                 # 定期実行の本体
.venv/bin/paper-summarizer status              # 処理状況
.venv/bin/paper-summarizer review <project>    # プロジェクトの文献レビュー (手動)
.venv/bin/paper-summarizer retry <id>          # failed を再キュー
.venv/bin/paper-summarizer try x.pdf --out /tmp/t [--model M]   # 試験要約 (モデル比較用)
.venv/bin/paper-summarizer test-mail
```

## セットアップ

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple
uv pip install --python .venv -e '.[dev]'
.venv/bin/docling-tools models download        # Docling のモデルを事前取得 (以後オフライン動作)
cp .env.example .env && chmod 600 .env         # Gmail アプリパスワードを設定

# 要 sudo
sudo mkdir -p /data/papers /var/log/paper-summarizer
sudo chown ito:ito /data/papers /var/log/paper-summarizer
sudo cp systemd/paper-summarizer.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now paper-summarizer.timer
```

### Syncthing (論文フォルダの共有)

```bash
sudo apt install syncthing
sudo systemctl enable --now syncthing@ito
echo ".state" > /data/papers/.stignore          # 状態 DB・作業領域は同期しない
```

- 管理画面は `http://127.0.0.1:8384` (Spark 上)。リモートから設定する場合は `ssh -L 8384:127.0.0.1:8384 spark` で転送する
- `/data/papers` を共有フォルダとして追加し、手元 PC の Syncthing とデバイス ID を交換して接続する
- 外部への流出防止のため、設定 → 接続 で **グローバルディスカバリ・リレーを無効化** し、LAN 内 (または VPN 経由) の直接接続のみにする

ログ: `/var/log/paper-summarizer/paper-summarizer.log` (日次ローテーション・180 日保持) と `journalctl -u paper-summarizer`。
LLM 呼び出しごとのトークン数・所要時間は SQLite の `llm_calls` テーブルに記録される。
