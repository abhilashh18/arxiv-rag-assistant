import argparse
import json
from pathlib import Path

from src.embeddings.embedder import generate_embeddings
from src.retrieval.faiss_index import (
    add_embeddings_to_index,
    create_index,
    load_index,
    save_index,
)

CHUNKS_PATH = Path("data/processed/all_chunks.json")
INDEX_PATH = Path("data/index/faiss.index")
INDEXED_IDS_PATH = Path("data/index/indexed_chunk_ids.json")


def _load_indexed_ids():
    if INDEXED_IDS_PATH.exists():
        with open(INDEXED_IDS_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_indexed_ids(indexed_ids):
    INDEXED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEXED_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(indexed_ids), f)


def build_index(rebuild=False):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    indexed_ids = set() if rebuild else _load_indexed_ids()

    # Only embed chunks we haven't already indexed. This is safe BECAUSE
    # chunking.py now guarantees chunk_ids are stable and append-only
    # across incremental runs -- an existing chunk's position never
    # changes, so merging new vectors into the existing FAISS index
    # (which stores vectors by insertion order) stays consistent with
    # this file's chunk order.
    chunks_to_embed = [c for c in chunks if c["chunk_id"] not in indexed_ids]

    if not rebuild and INDEX_PATH.exists() and not chunks_to_embed:
        print("Index is already up to date -- nothing new to embed.")
        return

    if rebuild or not INDEX_PATH.exists():
        print(f"Building index from scratch ({len(chunks)} chunks)...")
        embeddings = generate_embeddings(chunks, show_progress=True)
        index = create_index(embeddings)
        indexed_ids = {c["chunk_id"] for c in chunks}
    else:
        print(f"Embedding {len(chunks_to_embed)} new chunk(s) "
              f"(skipping {len(chunks) - len(chunks_to_embed)} already indexed)...")
        embeddings = generate_embeddings(chunks_to_embed, show_progress=True)
        index = load_index(str(INDEX_PATH))
        index = add_embeddings_to_index(index, embeddings)
        indexed_ids |= {c["chunk_id"] for c in chunks_to_embed}

    print(f"Index now contains {index.ntotal} vector(s).")

    save_index(index, str(INDEX_PATH))
    _save_indexed_ids(indexed_ids)

    print("FAISS index saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build or update the FAISS index from chunked papers."
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild the entire index from scratch instead of only "
             "embedding new chunks (use after changing the embedding "
             "model or fixing a chunking bug).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(rebuild=args.rebuild)