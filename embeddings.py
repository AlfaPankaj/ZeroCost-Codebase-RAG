import numpy as np

_model_instance = None

def _get_model():
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading local embedding model... (only happens once)")
            _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("ERROR: sentence-transformers is not installed.")
            raise
    return _model_instance

def embed_text(text: str) -> np.ndarray:
    """Get embedding vector for a piece of text using sentence-transformers."""
    try:
        model = _get_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(384, dtype=np.float32)

def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of texts rapidly with a progress bar."""
    if not texts:
        return []
    try:
        model = _get_model()
        embeddings = model.encode(
            texts, 
            batch_size=128, 
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        return list(embeddings)
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return [np.zeros(384, dtype=np.float32) for _ in texts]
