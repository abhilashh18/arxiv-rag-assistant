import json
from pathlib import Path

from src.embeddings.embedder import generate_embeddings
from src.retrieval.faiss_index import load_index

CHUNKS_PATH = Path("data/processed/all_chunks.json")
INDEX_PATH = Path("data/index/faiss.index")

# Module-level cache for lazily-loaded chunks/index. Deliberately NOT
# loaded at import time -- importing this module (e.g. for a quick unit
# test) should be instant and should never fail just because the index
# hasn't been built yet.
_chunks = None
_index = None


def _load_retrieval_state():
    global _chunks, _index

    if _chunks is not None and _index is not None:
        return _chunks, _index

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `python -m src.chunking.chunking` "
            "first to generate chunks from your downloaded papers."
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} not found. Run `python -m src.retrieval.build_index` "
            "first to build the search index from your chunks."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = load_index(str(INDEX_PATH))

    # The retriever assumes chunks[i] corresponds to the i-th vector
    # FAISS indexed (chunking.py/build_index.py guarantee this by only
    # ever appending, never reordering). Check the one thing that's
    # cheap and would catch the most common way that invariant breaks --
    # a stale index built before the most recent chunks were added, or
    # vice versa -- rather than silently returning wrong results.
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunks mismatch: index has {index.ntotal} vectors but "
            f"{CHUNKS_PATH} has {len(chunks)} chunks. Run "
            "`python -m src.retrieval.build_index` to bring them back in sync."
        )

    _chunks, _index = chunks, index
    return _chunks, _index


def retrieve_chunks(query, top_k=5):
    query = query.strip()
    if not query:
        return []

    import faiss

    chunks, index = _load_retrieval_state()

    query_embedding = generate_embeddings([
        {"text": query}
    ]).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS pads with -1 when top_k > number of vectors
        results.append({
            "chunk": chunks[idx],
            "score": float(distance)
        })

    return results