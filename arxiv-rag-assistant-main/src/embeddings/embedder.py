MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache for the lazily-loaded model. Deliberately NOT loaded
# at import time -- importing this module should be near-instant. Both
# the model weights AND the heavy torch/sentence-transformers library
# imports themselves are deferred to first real use.
_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{MODEL_NAME}' on {device}...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def generate_embeddings(chunks, batch_size=32, show_progress=False):
    model = _get_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )

    return embeddings