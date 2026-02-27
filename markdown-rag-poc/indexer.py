"""
indexer.py - Markdownファイルを読み込み、チャンク分割してChromaDBに保存する
"""

import os
import re
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

COLLECTION_NAME = "markdown_docs"
CHUNK_SIZE = 500  # 見出しがない場合のフォールバックサイズ（文字数）
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100  # OpenAI API への1回あたりの最大テキスト数


def find_markdown_files(directory: str) -> list[Path]:
    """指定ディレクトリ以下の .md ファイルを再帰的に収集する"""
    root = Path(directory)
    if not root.exists():
        raise ValueError(f"Directory not found: {directory}")
    files = sorted(root.rglob("*.md"))
    return files


def split_by_headings(text: str) -> list[str]:
    """
    見出し（# ～ ######）単位でテキストを分割する。
    見出しがない場合は CHUNK_SIZE 文字で分割する。
    """
    # 行頭の見出しで分割（見出し自体をチャンク先頭に含める）
    parts = re.split(r"(?=\n#{1,6} )", text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 見出しを含むチャンクはそのまま追加
        if re.match(r"#{1,6} ", part):
            chunks.append(part)
        else:
            # 見出しなし → CHUNK_SIZE 文字ごとに分割
            for i in range(0, len(part), CHUNK_SIZE):
                chunk = part[i : i + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)

    return chunks


def extract_heading(chunk: str) -> str:
    """チャンク先頭の見出しテキストを抽出する（なければ空文字）"""
    match = re.match(r"#{1,6} (.+)", chunk)
    return match.group(1).strip() if match else ""


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """テキストのリストをEmbeddingに変換する（バッチ処理）"""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def build_index(directory: str, db_path: str = "./chroma_db") -> None:
    """
    指定ディレクトリの Markdown ファイルを読み込み、
    チャンク分割 → Embedding → ChromaDB に保存する。
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=db_path)

    # 既存コレクションがあれば削除して再作成（再インデックス）
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"[indexer] Deleted existing collection: {COLLECTION_NAME}")

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = find_markdown_files(directory)
    if not md_files:
        print(f"[indexer] No .md files found in: {directory}")
        return

    print(f"[indexer] Found {len(md_files)} markdown file(s)")

    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for md_file in md_files:
        rel_path = str(md_file)
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[indexer] Warning: could not read {rel_path}: {e}")
            continue

        chunks = split_by_headings(text)
        print(f"[indexer]   {rel_path}: {len(chunks)} chunk(s)")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::{idx}"
            heading = extract_heading(chunk)
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append({"source": rel_path, "heading": heading, "chunk_index": idx})

    if not all_documents:
        print("[indexer] No content to index.")
        return

    print(f"[indexer] Embedding {len(all_documents)} chunk(s) ...")
    embeddings = embed_texts(openai_client, all_documents)

    print("[indexer] Saving to ChromaDB ...")
    collection.add(
        ids=all_ids,
        documents=all_documents,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print(f"[indexer] Done. {len(all_documents)} chunk(s) indexed into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <markdown_directory> [db_path]")
        sys.exit(1)

    target_dir = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./chroma_db"
    build_index(target_dir, db_path)
