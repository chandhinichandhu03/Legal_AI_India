"""
dp_similarity.py — Dynamic Programming Algorithms for Legal Text Similarity
=============================================================================
LCS, Edit Distance, Sequence Alignment, Jaccard, Cosine Similarity,
Document Similarity Scoring with memoization.
"""

import re
import math
import numpy as np
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
from collections import Counter

from config import DP_CACHE_SIZE, SIMILARITY_THRESHOLD
from utils import logger, memoize


# ── Longest Common Subsequence ────────────────────────────────────────────────

def lcs_length(s1: str, s2: str) -> int:
    """Compute LCS length between two strings (token-level)."""
    tokens1 = s1.lower().split()
    tokens2 = s2.lower().split()

    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0

    # DP table with rolling array for memory efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def lcs_similarity(s1: str, s2: str) -> float:
    """LCS-based similarity score normalized by max length."""
    if not s1 or not s2:
        return 0.0
    lcs_len = lcs_length(s1, s2)
    max_len = max(len(s1.split()), len(s2.split()))
    return round(lcs_len / max_len, 4) if max_len > 0 else 0.0


def lcs_retrieve(s1: str, s2: str) -> str:
    """Retrieve the actual LCS tokens between two strings."""
    tokens1 = s1.lower().split()
    tokens2 = s2.lower().split()
    m, n = len(tokens1), len(tokens2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if tokens1[i - 1] == tokens2[j - 1]:
            result.append(tokens1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return " ".join(reversed(result))


# ── Edit Distance (Levenshtein) ───────────────────────────────────────────────

def edit_distance(s1: str, s2: str, token_level: bool = True) -> int:
    """Compute edit distance between two strings."""
    if token_level:
        seq1 = s1.lower().split()
        seq2 = s2.lower().split()
    else:
        seq1 = list(s1)
        seq2 = list(s2)

    m, n = len(seq1), len(seq2)

    # DP with rolling rows
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],      # deletion
                                   curr[j - 1],   # insertion
                                   prev[j - 1])   # substitution
        prev = curr

    return prev[n]


def edit_distance_similarity(s1: str, s2: str) -> float:
    """Normalized edit distance similarity (1 - normalized_distance)."""
    if not s1 or not s2:
        return 0.0
    tokens1 = s1.split()
    tokens2 = s2.split()
    dist = edit_distance(s1, s2)
    max_len = max(len(tokens1), len(tokens2))
    if max_len == 0:
        return 1.0
    return round(1.0 - dist / max_len, 4)


# ── Longest Increasing Subsequence ───────────────────────────────────────────

def lis_length(arr: List[int]) -> int:
    """Compute LIS length using patience sorting O(n log n)."""
    if not arr:
        return 0

    import bisect
    tails = []
    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


def section_order_similarity(sections1: List[str], sections2: List[str]) -> float:
    """Compare if legal sections appear in same order (LIS-based)."""
    if not sections1 or not sections2:
        return 0.0

    # Convert sections2 to index map
    idx_map = {s: i for i, s in enumerate(sections2)}
    arr = [idx_map[s] for s in sections1 if s in idx_map]

    if not arr:
        return 0.0

    lis_len = lis_length(arr)
    return round(lis_len / max(len(sections1), len(sections2)), 4)


# ── Smith-Waterman Sequence Alignment ────────────────────────────────────────

def sequence_alignment(s1: str, s2: str, match: int = 2, mismatch: int = -1,
                        gap: int = -1) -> Tuple[float, str, str]:
    """Local sequence alignment (Smith-Waterman) for legal text matching."""
    tokens1 = s1.lower().split()[:100]  # Limit for performance
    tokens2 = s2.lower().split()[:100]
    m, n = len(tokens1), len(tokens2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_score = 0
    max_pos = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = dp[i - 1][j - 1] + (match if tokens1[i - 1] == tokens2[j - 1] else mismatch)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            dp[i][j] = max(0, diag, up, left)
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)

    # Normalize score
    max_possible = match * min(m, n)
    score = round(max_score / max_possible, 4) if max_possible > 0 else 0.0

    return score, s1[:50], s2[:50]


# ── Jaccard Similarity ────────────────────────────────────────────────────────

def jaccard_similarity(s1: str, s2: str, n_gram: int = 1) -> float:
    """Jaccard similarity on token n-grams."""
    def get_ngrams(text: str, n: int) -> set:
        tokens = text.lower().split()
        return set(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    set1 = get_ngrams(s1, n_gram)
    set2 = get_ngrams(s2, n_gram)

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union, 4) if union > 0 else 0.0


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def cosine_similarity_text(s1: str, s2: str) -> float:
    """Cosine similarity using TF vectors."""
    def tf_vector(text: str) -> Dict[str, float]:
        tokens = text.lower().split()
        freq = Counter(tokens)
        total = len(tokens)
        return {t: c / total for t, c in freq.items()}

    v1 = tf_vector(s1)
    v2 = tf_vector(s2)

    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0

    dot = sum(v1[t] * v2[t] for t in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))

    if mag1 * mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 4)


# ── Document Similarity Score (Composite) ────────────────────────────────────

def document_similarity_score(doc1: str, doc2: str) -> Dict:
    """
    Compute composite similarity score combining multiple DP algorithms.
    Returns detailed breakdown for explainability.
    """
    # Truncate for performance
    d1 = doc1[:3000] if doc1 else ""
    d2 = doc2[:3000] if doc2 else ""

    if not d1 or not d2:
        return {"total_score": 0.0, "breakdown": {}}

    lcs_sim = lcs_similarity(d1, d2)
    edit_sim = edit_distance_similarity(d1, d2)
    jacc_sim = jaccard_similarity(d1, d2)
    cos_sim = cosine_similarity_text(d1, d2)
    align_score, _, _ = sequence_alignment(d1, d2)

    # Weighted composite
    weights = {
        "lcs": 0.25,
        "edit_distance": 0.15,
        "jaccard": 0.25,
        "cosine": 0.25,
        "alignment": 0.10,
    }

    breakdown = {
        "lcs": lcs_sim,
        "edit_distance": edit_sim,
        "jaccard": jacc_sim,
        "cosine": cos_sim,
        "alignment": align_score,
    }

    total = sum(weights[k] * v for k, v in breakdown.items())

    return {
        "total_score": round(total, 4),
        "breakdown": breakdown,
        "is_similar": total >= SIMILARITY_THRESHOLD,
    }


# ── Find Most Similar Case ───────────────────────────────────────────────────

def find_most_similar(query: str, candidates: List[Dict],
                       text_field: str = "content",
                       top_k: int = 5) -> List[Dict]:
    """
    Find the most similar cases from a list of candidate documents.
    Uses memoized composite similarity scoring.
    Returns ranked list with scores.
    """
    results = []
    for doc in candidates:
        text = doc.get(text_field, "")
        if not text:
            continue
        score_data = document_similarity_score(query, text)
        results.append({
            **doc,
            "similarity_score": score_data["total_score"],
            "score_breakdown": score_data["breakdown"],
            "is_similar": score_data["is_similar"],
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


# ── Memoized LCS for repeated searches ───────────────────────────────────────

@lru_cache(maxsize=DP_CACHE_SIZE)
def cached_lcs(s1: str, s2: str) -> float:
    """Memoized LCS similarity for repeated pair comparisons."""
    return lcs_similarity(s1, s2)


@lru_cache(maxsize=DP_CACHE_SIZE)
def cached_edit_distance(s1: str, s2: str) -> float:
    """Memoized edit distance similarity."""
    return edit_distance_similarity(s1, s2)


@lru_cache(maxsize=DP_CACHE_SIZE)
def cached_jaccard(s1: str, s2: str) -> float:
    """Memoized Jaccard similarity."""
    return jaccard_similarity(s1, s2)
