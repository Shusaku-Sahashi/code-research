# Markdown RAG PoC - レポート

## 概要

MarkdownファイルをRAG（Retrieval-Augmented Generation）で検索・回答するPoCを実装した。
LangChainを使わずシンプルな直接API呼び出しで構築し、ChromaDB + OpenAI Embedding + Claude Haikuを組み合わせた。

---

## ファイル構成

```
markdown-rag-poc/
├── README.md         # このレポート
├── notes.md          # 実装メモ
├── requirements.txt  # 依存パッケージ
├── indexer.py        # インデックス作成（MD読込 → チャンク → Embedding → ChromaDB）
├── query.py          # 検索・回答（質問 → ChromaDB検索 → LLM回答生成）
└── main.py           # CLIエントリーポイント（index/ask/chatサブコマンド）
```

---

## アーキテクチャ

```
[Markdownファイル群]
       ↓ find_markdown_files() で再帰収集
[テキスト]
       ↓ split_by_headings() でチャンク分割
[チャンク + メタデータ（ファイルパス・見出し）]
       ↓ OpenAI text-embedding-3-small でEmbedding
[ベクトル]
       ↓ ChromaDB（PersistentClient）に保存
```

```
[ユーザー質問]
       ↓ OpenAI text-embedding-3-small でEmbedding
[質問ベクトル]
       ↓ ChromaDB cosine類似度検索（top-k件）
[関連チャンク + ソースファイル名]
       ↓ claude-haiku-4-5 にコンテキスト + 質問を渡す
[回答テキスト + ソースファイル一覧]
```

---

## 技術仕様

| 項目 | 選択 | 理由 |
|------|------|------|
| ベクトルDB | ChromaDB PersistentClient | ローカルで完結、永続化が簡単 |
| Embedding | text-embedding-3-small | 低コスト・高速・1536次元 |
| LLM | claude-haiku-4-5-20251001 | 高速・低コスト、日本語対応 |
| チャンク分割 | 見出し単位（フォールバック500文字） | 文脈の保持と検索精度のバランス |

---

## チャンク分割の詳細

```python
# 正規表現で行頭の見出しを検出して分割
re.split(r"(?=\n#{1,6} )", text)
```

- `# Section A` ～ `###### Section F` を全レベルで検出
- 見出しを含むチャンクはそのまま1チャンク
- 見出しなしテキストは500文字ずつ分割
- 各チャンクのメタデータ: `source`（ファイルパス）、`heading`（見出し文字列）、`chunk_index`

---

## ChromaDB の設計

- コレクション名: `markdown_docs`（固定）
- 距離関数: `cosine`（コサイン類似度）
- ID形式: `{ファイルパス}::{チャンクインデックス}`（一意性を保証）
- 再インデックス時: 既存コレクションを削除して再作成

---

## 使い方

### 前提条件

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### インデックス作成

```bash
python main.py index /path/to/markdown/docs
# オプション: --db ./my_chroma_db  (デフォルト: ./chroma_db)
```

実行例:
```
[indexer] Found 42 markdown file(s)
[indexer]   /docs/README.md: 5 chunk(s)
[indexer]   /docs/api.md: 12 chunk(s)
...
[indexer] Embedding 183 chunk(s) ...
[indexer] Saving to ChromaDB ...
[indexer] Done. 183 chunk(s) indexed into 'markdown_docs'
```

### 一回質問 (ask)

```bash
python main.py ask "インストール方法を教えてください"
```

出力例:
```
質問: インストール方法を教えてください

=== 回答 ===
インストールは以下のコマンドで行います：

```bash
pip install <package>
```

=== 参照ソース ===
  - /docs/install.md > インストール手順
  - /docs/README.md > セットアップ
```

### 対話モード (chat)

```bash
python main.py chat
```

```
=== Markdown RAG チャットモード ===
終了するには 'exit' または 'quit' を入力してください。

質問: APIの使い方は？
=== 回答 ===
...
質問: exit
終了します。
```

---

## テスト結果

モックを用いたインテグレーションテストで以下を確認済み:

1. **チャンク分割テスト**
   - 見出しあり: 3セクション → 3チャンク（正常）
   - 見出しなし1200文字 → 500 / 500 / 200文字の3チャンク（正常）

2. **インデックス作成テスト（モック）**
   - OpenAI Embedding API呼び出しをモック化
   - 3チャンクが ChromaDB に正常保存されることを確認

3. **検索・回答テスト（モック）**
   - ChromaDB から正しいソース付きで検索されることを確認
   - Anthropic API が `claude-haiku-4-5-20251001` モデルで呼び出されることを確認
   - ソースが `ファイルパス > 見出し` 形式で返却されることを確認

---

## 設計上の判断

### LangChain を使わない理由
- RAGの基本フローは3ステップ（Embed → Search → Generate）であり、フレームワーク不要
- 依存が少なく、内部処理が明確で修正しやすい
- デバッグが容易

### 見出し単位のチャンク分割
- Markdownの構造（セクション）を尊重することで、意味のあるまとまりに分割
- フォールバックの500文字は OpenAI の推奨する適切なチャンクサイズに近い
- 見出しテキストをメタデータに保存し、ソース表示で活用

### ChromaDB のPersistentClient
- ローカルファイルに永続化するため、プロセス再起動後も検索可能
- 再インデックス時はコレクション削除→再作成でクリーンな状態を維持

---

## 今後の改善点（PoC卒業時）

- [ ] インクリメンタルインデックス（差分のみ更新）
- [ ] チャンクのオーバーラップ（前後の文脈を含める）
- [ ] 複数の類似度閾値フィルタリング
- [ ] 検索件数（top_k）の動的調整
- [ ] Web UI の追加（Streamlit等）
- [ ] マルチコレクション対応（プロジェクト別）
