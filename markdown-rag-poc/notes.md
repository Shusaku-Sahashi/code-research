# Notes - Markdown RAG PoC

## 概要
MarkdownファイルをRAG（Retrieval-Augmented Generation）で検索・回答するPoCを作成。

## 構成方針

### 技術選定
- **ChromaDB**: ローカルで動作するベクトルDB。永続化も可能
- **OpenAI text-embedding-3-small**: 軽量・高速・低コストなEmbeddingモデル
- **claude-haiku-4-5**: 高速・低コストなLLM。回答生成に使用
- **LangChain不使用**: シンプルさ優先で直接API呼び出し

### チャンク分割方針
- `#`, `##`, `###` などの見出し単位で分割
- 見出しがない場合は500文字程度で分割
- 各チャンクにメタデータ（ファイルパス、見出し）を付与

### ファイル構成
```
markdown-rag-poc/
├── notes.md           # このファイル
├── READ.md            # 最終レポート
├── pyproject.toml     # uv プロジェクト定義（依存関係管理）
├── docker-compose.yml # ChromaDB サーバー（オプション）
├── .env.example       # 環境変数サンプル
├── indexer.py         # MD読み込み → チャンク分割 → Embedding → ChromaDB保存
├── query.py           # 質問 → ChromaDB検索 → LLM回答生成
└── main.py            # CLIエントリーポイント
```

## 実装メモ

### ChromaDB の使い方
- `chromadb.PersistentClient(path=...)` でローカルに永続化
- コレクション名を固定して、indexとqueryで共通使用
- `collection.add()` でembedding + document + metadata を一括保存
- `collection.query()` でベクトル近傍検索

### チャンク分割ロジック
- `re.split(r'\n(?=#{1,6} )', text)` で見出し単位に分割
- 分割結果が空または短すぎる場合はスキップ
- メタデータにファイルパスと見出しタイトルを保存

### Embedding
- OpenAI API の `embeddings.create()` で取得
- バッチ処理でAPI呼び出し回数を最小化

### RAG フロー
1. 質問文をEmbeddingに変換
2. ChromaDBで上位N件を検索
3. 検索結果（コンテキスト）+ 質問をLLMに渡して回答生成
4. ソースファイル名も表示

## 試したこと / 学んだこと

### 2026-02-27
- ChromaDB は `chromadb>=0.4` からAPIが変わっている。`Client()` → `PersistentClient()`
- OpenAI Embedding のバッチ上限は8191トークン/テキスト
- チャンクが小さすぎると文脈が失われ、大きすぎるとembeddingの精度が下がる
- 見出しをチャンクの先頭に含めることで文脈を維持
- ChromaDBのIDは文字列で一意である必要がある → `{filepath}_{chunk_index}` 形式を採用
- uv に移行: `requirements.txt` → `pyproject.toml`。`uv sync` で環境構築、`uv run python main.py ...` で実行
- ChromaDB は 2 モードに対応:
  - **ローカル埋め込みモード** (デフォルト): `PersistentClient` で `./chroma_db` に保存。Docker 不要
  - **サーバーモード**: `docker compose up` で起動し、`CHROMA_HOST=localhost` を設定すると `HttpClient` を使用
- `get_chroma_client()` ヘルパーを `indexer.py` に定義し、`CHROMA_HOST` 環境変数でモードを切り替え
