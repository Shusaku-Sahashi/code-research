"""
query.py - ChromaDB を検索して LLM (claude-haiku-4-5) で回答を生成する
"""

import os

import anthropic
import chromadb
from openai import OpenAI

from indexer import COLLECTION_NAME, EMBED_MODEL

TOP_K = 5  # 検索で取得する上位件数
LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """あなたはMarkdownドキュメントのQAアシスタントです。
提供されたコンテキスト（Markdownの抜粋）に基づいて、ユーザーの質問に日本語で回答してください。

回答のルール：
- コンテキストに含まれる情報のみを使って回答する
- コンテキストに答えがない場合は「提供されたドキュメントには該当する情報が見つかりませんでした」と伝える
- 回答は簡潔・明確にする
- コードブロックや箇条書きを適切に使う"""


def build_context(results: dict) -> tuple[str, list[str]]:
    """ChromaDB の検索結果からコンテキスト文字列とソースファイル一覧を生成する"""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources: list[str] = []
    context_parts: list[str] = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        source = meta.get("source", "unknown")
        heading = meta.get("heading", "")

        # ソース表示用（重複排除）
        source_label = f"{source}" + (f" > {heading}" if heading else "")
        if source_label not in sources:
            sources.append(source_label)

        # コンテキスト構築
        header = f"[出典 {i+1}: {source_label} (類似度スコア: {1 - dist:.3f})]"
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def answer_question(
    question: str,
    db_path: str = "./chroma_db",
    top_k: int = TOP_K,
) -> tuple[str, list[str]]:
    """
    質問に対して:
    1. 質問をEmbeddingに変換
    2. ChromaDBで類似チャンクを検索
    3. LLMで回答を生成

    Returns:
        (回答テキスト, ソースファイルのリスト)
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=db_path)

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # 質問をEmbeddingに変換
    embed_response = openai_client.embeddings.create(model=EMBED_MODEL, input=[question])
    query_embedding = embed_response.data[0].embedding

    # ChromaDB で類似チャンクを検索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return "インデックスにドキュメントが見つかりませんでした。先にインデックスを作成してください。", []

    context, sources = build_context(results)

    # LLM に回答を生成させる
    user_message = f"""以下のコンテキストを参考に、質問に回答してください。

## コンテキスト

{context}

## 質問

{question}"""

    message = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = message.content[0].text
    return answer, sources


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python query.py <question> [db_path]")
        sys.exit(1)

    question = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./chroma_db"

    answer, sources = answer_question(question, db_path)
    print("\n=== 回答 ===")
    print(answer)
    print("\n=== ソース ===")
    for s in sources:
        print(f"  - {s}")
