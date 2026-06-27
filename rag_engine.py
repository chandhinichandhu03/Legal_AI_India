"""
rag_engine.py — Hybrid RAG (Retrieval Augmented Generation) Engine
====================================================================
Combines semantic (ChromaDB) + keyword (BM25) retrieval.
Query rewriting, re-ranking, multi-document reasoning.
"""

import re
import json
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math

from config import RETRIEVAL_TOP_K, RETRIEVAL_HYBRID_ALPHA, CHUNK_SIZE
from vector_store import retrieve_chunks, retrieve_by_keyword
from ollama_engine import (
    generate_response, generate_summary, generate_follow_up_questions,
    compute_confidence_score
)
from nlp_engine import extract_keywords
from utils import logger, generate_citation


# ── BM25 Keyword Retrieval ─────────────────────────────────────────────────────

class BM25:
    """BM25 ranking for keyword-based retrieval."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized_corpus = []
        self.idf = {}
        self.avgdl = 0

    def fit(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized_corpus) / max(len(self.tokenized_corpus), 1)

        # Compute IDF
        N = len(self.tokenized_corpus)
        df = defaultdict(int)
        for doc in self.tokenized_corpus:
            for term in set(doc):
                df[term] += 1

        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = query.lower().split()
        doc = self.tokenized_corpus[doc_idx]
        dl = len(doc)
        tf = defaultdict(int)
        for t in doc:
            tf[t] += 1

        score = 0.0
        for token in tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            tf_val = tf.get(token, 0)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / max(denominator, 1e-6)

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── Query Rewriting ───────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Expand and rewrite legal query for better retrieval.
    """
    # Add legal context keywords
    legal_expansions = {
        "ipc 302": "IPC Section 302 murder death punishment",
        "ipc 420": "IPC Section 420 cheating fraud dishonestly",
        "ipc 376": "IPC Section 376 rape sexual assault",
        "ipc 379": "IPC Section 379 theft stolen property",
        "section 125": "CrPC Section 125 maintenance wife children",
        "dv act": "Domestic Violence Protection of Women Act",
        "pocso": "POCSO Act Protection of Children Sexual Offences",
        "rti": "Right to Information Act 2005 public authority",
        "rera": "Real Estate Regulation Act builder developer",
        "consumer": "Consumer Protection Act 2019 complaint forum",
        "mact": "Motor Accident Claims Tribunal compensation",
        "esi": "Employees State Insurance Act worker",
        "epf": "Employees Provident Fund EPFO deduction",
    }

    q_lower = query.lower()
    expanded = query
    for key, expansion in legal_expansions.items():
        if key in q_lower:
            expanded = f"{query} {expansion}"
            break

    # Extract and add IPC sections
    sections = re.findall(r'(?:IPC|Section|Sec\.?)\s*(\d+[A-Z]?)', query, re.IGNORECASE)
    if sections:
        expanded += f" Indian Penal Code section {' '.join(sections)}"

    return expanded


def retrieve_hybrid(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    alpha: float = RETRIEVAL_HYBRID_ALPHA,
    doc_chunks: List[str] = None,
) -> List[Dict]:
    """
    Hybrid retrieval: alpha * semantic_score + (1-alpha) * bm25_score.

    Args:
        query: User query
        top_k: Number of results
        alpha: Weight for semantic vs keyword (0=keyword, 1=semantic)
        doc_chunks: Pre-loaded chunks for BM25 (optional)

    Returns:
        Ranked list of chunks with hybrid scores
    """
    expanded_query = rewrite_query(query)

    # Semantic retrieval
    semantic_chunks = retrieve_chunks(expanded_query, top_k=top_k * 2)

    # Keyword retrieval
    keywords = extract_keywords(query, top_n=5)
    keyword_query = " ".join(keywords) if keywords else query
    keyword_chunks = retrieve_by_keyword(keyword_query, top_k=top_k * 2)

    # Merge and re-rank
    all_chunks = {}

    for chunk in semantic_chunks:
        key = chunk.get("text", "")[:100]  # Use text as key
        if key not in all_chunks:
            all_chunks[key] = chunk.copy()
            all_chunks[key]["semantic_score"] = chunk["score"]
            all_chunks[key]["keyword_score"] = 0.0
        else:
            all_chunks[key]["semantic_score"] = chunk["score"]

    for chunk in keyword_chunks:
        key = chunk.get("text", "")[:100]
        if key not in all_chunks:
            all_chunks[key] = chunk.copy()
            all_chunks[key]["semantic_score"] = 0.0
            all_chunks[key]["keyword_score"] = chunk["score"]
        else:
            all_chunks[key]["keyword_score"] = chunk["score"]

    # Compute hybrid score
    results = []
    for key, chunk in all_chunks.items():
        sem = chunk.get("semantic_score", 0.0)
        kw = chunk.get("keyword_score", 0.0)
        hybrid = alpha * sem + (1 - alpha) * kw
        chunk["score"] = round(hybrid, 4)
        chunk["hybrid_score"] = round(hybrid, 4)
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Re-rank retrieved chunks using BM25 on the retrieved set.
    """
    if not chunks:
        return []

    texts = [c.get("text", "") for c in chunks]
    bm25 = BM25()
    bm25.fit(texts)

    scored = bm25.retrieve(query, top_k=len(texts))
    ranked = []
    for idx, bm25_score in scored:
        chunk = chunks[idx].copy()
        original_score = chunk.get("score", 0.0)
        # Combine: 60% original semantic + 40% BM25
        chunk["rerank_score"] = round(0.6 * original_score + 0.4 * min(bm25_score / 10, 1.0), 4)
        chunk["score"] = chunk["rerank_score"]
        ranked.append(chunk)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def build_context_from_chunks(chunks: List[Dict], max_chars: int = 2500) -> Tuple[str, List[Dict]]:
    """
    Build context string from retrieved chunks.
    Returns (context_string, source_list).
    Only includes chunks with a relevance score >= 0.25 to prevent noise.
    """
    context_parts = []
    sources = []
    total_chars = 0

    # Filter out low-relevance matches
    valid_chunks = [c for c in chunks if c.get("score", 0.0) >= 0.25]
    
    # If no chunks met the threshold, fallback to the top 1 match if it's at least 0.15
    if not valid_chunks and chunks and chunks[0].get("score", 0.0) >= 0.15:
        valid_chunks = [chunks[0]]

    for i, chunk in enumerate(valid_chunks):
        text = chunk.get("text", "")
        doc_title = chunk.get("doc_title", "Unknown Document")
        score = chunk.get("score", 0.0)

        if total_chars + len(text) > max_chars:
            break

        context_parts.append(f"[Source {i+1}: {doc_title} (relevance: {score:.2%})]\n{text}")
        total_chars += len(text)

        sources.append({
            "index": i + 1,
            "doc_title": doc_title,
            "doc_id": chunk.get("doc_id"),
            "score": score,
            "text_preview": text[:200],
        })

    return "\n\n".join(context_parts), sources


def answer_question(
    question: str,
    model: str = None,
    conversation_history: List[Dict] = None,
    language: str = "en",
) -> Dict:
    """
    Full RAG pipeline:
    1. Rewrite query
    2. Hybrid retrieval
    3. Re-rank
    4. Build context
    5. Generate answer with Ollama
    6. Compute confidence
    7. Generate follow-up questions
    8. Translate if needed
    """
    start_time = time.time()

    # Retrieve
    chunks = retrieve_hybrid(question, top_k=RETRIEVAL_TOP_K)
    chunks = rerank_chunks(question, chunks)

    # Build context
    context, sources = build_context_from_chunks(chunks)

    # Generate
    result = generate_response(
        question=question,
        context=context,
        model=model,
        conversation_history=conversation_history,
    )

    answer = result.get("answer", "")
    confidence = compute_confidence_score(chunks, answer)

    # Citations
    citations = []
    for source in sources:
        citation = generate_citation(
            doc_title=source["doc_title"],
            page=None,
        )
        citations.append(citation)

    # Follow-up questions
    follow_ups = []
    if answer and len(answer) > 50:
        try:
            follow_ups = generate_follow_up_questions(question, answer, model)
        except Exception:
            pass

    # Translation
    translated_answer = None
    if language and language != "en":
        try:
            from translator import translate_text
            t_result = translate_text(answer, target_lang=language)
            translated_answer = t_result.get("translated", answer)
        except Exception:
            pass

    total_time = time.time() - start_time

    return {
        "question": question,
        "answer": answer,
        "translated_answer": translated_answer,
        "language": language,
        "sources": sources,
        "context": context[:1000],  # Preview
        "confidence_score": confidence,
        "hallucination_score": result.get("hallucination_score", 0.0),
        "hallucination_flag": result.get("hallucination_flag", False),
        "citations": citations,
        "follow_up_questions": follow_ups,
        "model_used": result.get("model", model),
        "response_time": round(total_time, 2),
        "tokens_used": result.get("tokens_used", 0),
        "chunks_retrieved": len(chunks),
        "success": result.get("success", True),
    }
