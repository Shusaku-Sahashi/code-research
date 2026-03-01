"""
indexer.py - Load Markdown files, split into chunks, and store in ChromaDB.
"""

import os
import re
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

COLLECTION_NAME = "markdown_docs"
CHUNK_SIZE = 500  # Fallback chunk size in characters when no headings are found
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100  # Max number of texts per OpenAI API call


def get_chroma_client(db_path: str = "./chroma_db") -> chromadb.ClientAPI:
    """
    Return a ChromaDB client.

    - If CHROMA_HOST is set, connect to a remote ChromaDB server via HttpClient
      (use with `docker compose up`).
    - Otherwise, use a local PersistentClient stored at db_path.
    """
    host = os.environ.get("CHROMA_HOST")
    if host:
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        print(f"[chroma] Connecting to ChromaDB server at {host}:{port}")
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.PersistentClient(path=db_path)


def find_markdown_files(directory: str) -> list[Path]:
    """Recursively collect all .md files under the given directory."""
    root = Path(directory)
    if not root.exists():
        raise ValueError(f"Directory not found: {directory}")
    files = sorted(root.rglob("*.md"))
    return files


def split_by_headings(text: str) -> list[str]:
    """
    Split text by Markdown headings (# through ######).
    Falls back to CHUNK_SIZE character splits when no headings are present.
    """
    # Split at line-leading headings, keeping the heading at the start of each chunk
    parts = re.split(r"(?=\n#{1,6} )", text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"#{1,6} ", part):
            # Chunk starts with a heading — keep as-is
            chunks.append(part)
        else:
            # No heading — split by CHUNK_SIZE characters
            for i in range(0, len(part), CHUNK_SIZE):
                chunk = part[i : i + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)

    return chunks


def extract_heading(chunk: str) -> str:
    """Extract the heading text from the first line of a chunk (empty string if none)."""
    match = re.match(r"#{1,6} (.+)", chunk)
    return match.group(1).strip() if match else ""


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Convert a list of texts to embeddings using batched API calls."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def build_index(directory: str, db_path: str = "./chroma_db") -> None:
    """
    Load Markdown files from the given directory, split into chunks,
    embed them, and store in ChromaDB.
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = get_chroma_client(db_path)

    # Drop existing collection to allow clean re-indexing
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
