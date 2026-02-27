"""
query.py - Search ChromaDB and generate answers using claude-haiku-4-5.

Supports multi-query expansion: the original question is rewritten into
N alternative phrasings, each searched independently, and results are merged
before passing to the LLM.
"""

import os

import anthropic
import chromadb
from openai import OpenAI

from indexer import COLLECTION_NAME, EMBED_MODEL, get_chroma_client

TOP_K = 5          # Number of top results to retrieve
N_EXPANSIONS = 3   # Number of alternative queries to generate
LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """あなたはMarkdownドキュメントのQAアシスタントです。
提供されたコンテキスト（Markdownの抜粋）に基づいて、ユーザーの質問に日本語で回答してください。

回答のルール：
- コンテキストに含まれる情報のみを使って回答する
- コンテキストに答えがない場合は「提供されたドキュメントには該当する情報が見つかりませんでした」と伝える
- 回答は簡潔・明確にする
- コードブロックや箇条書きを適切に使う"""

EXPANSION_PROMPT = """\
Generate {n} alternative phrasings of the following question to improve \
document retrieval. Each phrasing should approach the same information need \
from a different angle. Output only the questions, one per line, with no \
numbering or extra text.

Question: {question}"""


def expand_query(
    question: str,
    client: anthropic.Anthropic,
    n: int = N_EXPANSIONS,
) -> list[str]:
    """
    Use an LLM to generate N alternative phrasings of the question.
    Returns a list starting with the original question followed by expansions.
    """
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": EXPANSION_PROMPT.format(n=n, question=question)}],
    )
    lines = [line.strip() for line in message.content[0].text.strip().splitlines()]
    expansions = [line for line in lines if line and line != question]
    queries = [question] + expansions[:n]
    print(f"[query] Expanded into {len(queries)} queries:")
    for i, q in enumerate(queries):
        prefix = "  original" if i == 0 else f"  expand {i}"
        print(f"  {prefix}: {q}")
    return queries


def merge_results(results_list: list[dict], top_k: int) -> dict:
    """
    Merge and deduplicate ChromaDB results from multiple queries.
    For duplicate chunks (same ID), keep the best (lowest) distance.
    Returns results sorted by distance, capped at top_k.
    """
    # id -> (document, metadata, distance)
    best: dict[str, tuple[str, dict, float]] = {}

    for results in results_list:
        if not results["ids"] or not results["ids"][0]:
            continue
        for id_, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if id_ not in best or dist < best[id_][2]:
                best[id_] = (doc, meta, dist)

    sorted_items = sorted(best.values(), key=lambda x: x[2])[:top_k]
    if not sorted_items:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    docs, metas, dists = zip(*sorted_items)
    return {
        "ids": [[]],  # not needed downstream
        "documents": [list(docs)],
        "metadatas": [list(metas)],
        "distances": [list(dists)],
    }


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
    expand: bool = True,
    n_expansions: int = N_EXPANSIONS,
) -> tuple[str, list[str]]:
    """
    Answer a question using RAG with optional multi-query expansion:
    1. (Optional) Expand the question into multiple phrasings via LLM
    2. Embed each query and search ChromaDB
    3. Merge and deduplicate results, keeping the best similarity per chunk
    4. Generate an answer with the LLM using the merged context

    Returns:
        (answer text, list of source file labels)
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chroma_client = get_chroma_client(db_path)

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    n_docs = collection.count()

    # Build the list of queries to run
    if expand:
        queries = expand_query(question, anthropic_client, n=n_expansions)
    else:
        queries = [question]

    # Embed all queries in a single API call
    embed_response = openai_client.embeddings.create(model=EMBED_MODEL, input=queries)
    embeddings = [item.embedding for item in embed_response.data]

    # Search ChromaDB for each query (ids are always returned by ChromaDB)
    results_list = []
    for embedding in embeddings:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, n_docs),
            include=["documents", "metadatas", "distances"],
        )
        # ids are always returned by ChromaDB even without explicit include
        results_list.append(results)

    merged = merge_results(results_list, top_k)

    if not merged["documents"] or not merged["documents"][0]:
        return "インデックスにドキュメントが見つかりませんでした。先にインデックスを作成してください。", []

    context, sources = build_context(merged)

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
