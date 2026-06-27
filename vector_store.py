"""
vector_store.py — ChromaDB Vector Store Operations
====================================================
create_embeddings, store_embeddings, retrieve_chunks,
delete_vectors, update_vectors.
"""

import json
import uuid
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K
from embeddings import embed_texts, embed_single
from utils import logger

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        try:
            _client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            logger.info(f"ChromaDB initialized at {CHROMA_DIR}")
        except Exception as e:
            logger.error(f"ChromaDB init error: {e}")
            raise
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        try:
            _collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "description": "Indian Legal Documents"},
            )
            logger.info(f"ChromaDB collection ready: {CHROMA_COLLECTION_NAME} ({_collection.count()} docs)")
        except Exception as e:
            logger.error(f"ChromaDB collection error: {e}")
            raise
    return _collection


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings and return as list of lists."""
    embeddings = embed_texts(texts)
    return embeddings.tolist()


def store_embeddings(
    texts: List[str],
    doc_id: int,
    doc_title: str,
    metadatas: List[Dict] = None,
    ids: List[str] = None,
) -> int:
    """Store text chunks with embeddings in ChromaDB."""
    if not texts:
        return 0

    collection = _get_collection()

    # Generate IDs
    if ids is None:
        ids = [f"doc_{doc_id}_chunk_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(texts))]

    # Generate metadata
    if metadatas is None:
        metadatas = [{"doc_id": doc_id, "doc_title": doc_title, "chunk_idx": i}
                     for i in range(len(texts))]
    else:
        for i, m in enumerate(metadatas):
            m["doc_id"] = doc_id
            m["doc_title"] = doc_title
            m["chunk_idx"] = i

    # Generate embeddings
    embeddings = create_embeddings(texts)

    # Store in batches
    batch_size = 50
    stored = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        try:
            collection.add(
                documents=batch_texts,
                embeddings=batch_embs,
                ids=batch_ids,
                metadatas=batch_meta,
            )
            stored += len(batch_texts)
        except Exception as e:
            logger.error(f"ChromaDB store error (batch {i}): {e}")

    logger.info(f"Stored {stored} chunks for doc_id={doc_id}")
    return stored


def retrieve_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filter_doc_id: int = None,
    filter_case_type: str = None,
) -> List[Dict]:
    """
    Retrieve top-k similar chunks for a query.
    Returns list of {text, score, doc_id, doc_title, metadata}.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_emb = embed_single(query)

    # Build where filter
    where = {}
    if filter_doc_id:
        where["doc_id"] = filter_doc_id
    if filter_case_type:
        where["case_type"] = filter_case_type

    try:
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, collection.count()),
            where=where if where else None,
            include=["documents", "distances", "metadatas"],
        )
    except Exception as e:
        logger.error(f"ChromaDB retrieve error: {e}")
        return []

    chunks = []
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, dist, meta in zip(docs, dists, metas):
        similarity = 1 - dist  # Convert distance to similarity
        chunks.append({
            "text": doc,
            "score": round(float(similarity), 4),
            "doc_id": meta.get("doc_id"),
            "doc_title": meta.get("doc_title", ""),
            "chunk_idx": meta.get("chunk_idx", 0),
            "metadata": meta,
        })

    return sorted(chunks, key=lambda x: x["score"], reverse=True)


def retrieve_by_keyword(keyword: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    """Retrieve chunks containing a specific keyword."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    try:
        query_emb = embed_single(keyword)
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, collection.count()),
            include=["documents", "distances", "metadatas"],
        )
        chunks = []
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, dist, meta in zip(docs, dists, metas):
            chunks.append({
                "text": doc,
                "score": round(1 - float(dist), 4),
                "doc_id": meta.get("doc_id"),
                "doc_title": meta.get("doc_title", ""),
                "metadata": meta,
            })
        return chunks
    except Exception as e:
        logger.error(f"Keyword retrieval error: {e}")
        return []


def delete_vectors(doc_id: int) -> bool:
    """Delete all vectors associated with a document ID."""
    collection = _get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted vectors for doc_id={doc_id}")
        return True
    except Exception as e:
        logger.error(f"Delete vectors error: {e}")
        return False


def update_vectors(doc_id: int, texts: List[str], doc_title: str) -> int:
    """Update vectors by deleting old ones and inserting new."""
    delete_vectors(doc_id)
    return store_embeddings(texts, doc_id, doc_title)


def get_collection_stats() -> Dict:
    """Return ChromaDB collection statistics."""
    try:
        collection = _get_collection()
        return {
            "name": CHROMA_COLLECTION_NAME,
            "count": collection.count(),
            "path": str(CHROMA_DIR),
        }
    except Exception as e:
        return {"error": str(e)}
