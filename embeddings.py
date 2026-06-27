"""
embeddings.py — SentenceTransformers embedding generation
==========================================================
"""

import numpy as np
from typing import List, Union
from functools import lru_cache

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE
from utils import logger

_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _model


def embed_texts(texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE,
                show_progress: bool = False) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = _load_model()
    if not texts:
        return np.array([])

    # Clean texts
    texts = [t.strip() if t else "" for t in texts]
    texts = [t[:512] if len(t) > 512 else t for t in texts]  # Limit tokens

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise


def embed_single(text: str) -> np.ndarray:
    """Generate embedding for a single text."""
    result = embed_texts([text])
    return result[0] if len(result) > 0 else np.zeros(384)


def cosine_similarity_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if emb1 is None or emb2 is None:
        return 0.0
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding model output."""
    try:
        model = _load_model()
        return model.get_sentence_embedding_dimension()
    except Exception:
        return 384  # Default for all-MiniLM-L6-v2


def model_info() -> dict:
    """Return info about the loaded embedding model."""
    return {
        "model_name": EMBEDDING_MODEL,
        "device": EMBEDDING_DEVICE,
        "dimension": get_embedding_dimension(),
    }
