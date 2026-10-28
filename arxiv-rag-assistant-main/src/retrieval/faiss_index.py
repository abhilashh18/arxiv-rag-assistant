import os
import tempfile

import faiss
import numpy as np


def create_index(embeddings):
    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index(index, path):
    """
    Write the index atomically: write to a temp file in the same
    directory, then rename it over the target path. A crash or Ctrl-C
    mid-write leaves the temp file corrupted, but the real target path
    is never touched until the write has fully succeeded -- so the
    last known-good index is never destroyed by a failed save.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)

    try:
        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_index(path):
    return faiss.read_index(path)


def add_embeddings_to_index(index, new_embeddings):
    """Merge new vectors into an existing index in place (in-place normalize + add)."""
    new_embeddings = new_embeddings.astype("float32")
    faiss.normalize_L2(new_embeddings)
    index.add(new_embeddings)
    return index