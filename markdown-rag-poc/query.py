"""
query.py - Search ChromaDB and generate answers using claude-haiku-4-5.
"""

import os

import anthropic
import chromadb
from openai import OpenAI

from indexer import COLLECTION_NAME, EMBED_MODEL

TOP_K = 5  # Number of top results to retrieve
LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """あなたはMarkdownドキュメントのQAアシスタントです。
提供されたコンテキスト（Markdownの抜粋）に基づいて、ユーザーの質問に日本語で回答してください。

回答のルール：
- コンテキストに含まれる情報のみを使って回答する
- コンテキストに答えがない場合は「提供されたドキュメントには該当する情報が見つかりませんでした」と伝える
- 回答は簡潔・明確にする
- コードブロックや箇条書きを適切に使う"""


def build_context(results: dict) -> tuple[str, list[str]]:
    """Build a context string and deduplicated source list from ChromaDB query results."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources: list[str] = []
    context_parts: list[str] = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        source = meta.get("source", "unknown")
        heading = meta.get("heading", "")

        source_label = f"{source}" + (f" > {heading}" if heading else "")
        if source_label not in sources:
            sources.append(source_label)

        header = f"[Source {i+1}: {source_label} (similarity: {1 - dist:.3f})]"
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def answer_question(
    question: str,
    db_path: str = "./chroma_db",
    top_k: int = TOP_K,
) -> tuple[str, list[str]]:
    """
    Answer a question using RAG:
    1. Embed the question
    2. Search ChromaDB for similar chunks
    3. Generate an answer with the LLM

    Returns:
        (answer text, list of source file labels)
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=db_path)

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Embed the question
    embed_response = openai_client.embeddings.create(model=EMBED_MODEL, input=[question])
    query_embedding = embed_response.data[0].embedding

    # Search ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return "インデックスにドキュメントが見つかりませんでした。先にインデックスを作成してください。", []

    context, sources = build_context(results)

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
