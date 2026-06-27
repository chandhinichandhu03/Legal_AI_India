"""
search_engine.py — Advanced Legal Search Engine
================================================
Multi-field search with ranking: IPC section, case number,
keywords, judge name, date, act name, party name.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import ITEMS_PER_PAGE
from dp_similarity import cosine_similarity_text, jaccard_similarity
from utils import logger


def search_documents(
    query: str = "",
    ipc_section: str = "",
    judge_name: str = "",
    case_type: str = "",
    act_name: str = "",
    date_from: str = "",
    date_to: str = "",
    party_name: str = "",
    court: str = "",
    page: int = 1,
    per_page: int = ITEMS_PER_PAGE,
    sort_by: str = "relevance",
    owner_id: int = None,
) -> Dict:
    """
    Advanced multi-field document search.
    Returns paginated, ranked results.
    """
    try:
        from schema import Document
        from sqlalchemy import or_, and_, func

        q = Document.query

        # Owner filter
        if owner_id:
            q = q.filter(Document.owner_id == owner_id)

        # Case type filter
        if case_type:
            q = q.filter(Document.case_type == case_type)

        # IPC section filter
        if ipc_section:
            q = q.filter(Document.ipc_sections.contains(ipc_section))

        # Judge name filter
        if judge_name:
            q = q.filter(Document.judges.ilike(f"%{judge_name}%"))

        # Party name filter
        if party_name:
            q = q.filter(
                or_(
                    Document.parties.ilike(f"%{party_name}%"),
                    Document.title.ilike(f"%{party_name}%"),
                )
            )

        # Court filter
        if court:
            q = q.filter(Document.courts.ilike(f"%{court}%"))

        # Text query across multiple fields
        if query:
            search_term = f"%{query}%"
            q = q.filter(
                or_(
                    Document.title.ilike(search_term),
                    Document.content.ilike(search_term),
                    Document.summary.ilike(search_term),
                    Document.keywords.ilike(search_term),
                )
            )

        # Date range
        if date_from:
            try:
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                q = q.filter(Document.created_at >= date_from_dt)
            except ValueError:
                pass

        if date_to:
            try:
                date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                q = q.filter(Document.created_at <= date_to_dt)
            except ValueError:
                pass

        # Total count
        total = q.count()

        # Sorting
        if sort_by == "date_desc":
            q = q.order_by(Document.created_at.desc())
        elif sort_by == "date_asc":
            q = q.order_by(Document.created_at.asc())
        elif sort_by == "title":
            q = q.order_by(Document.title.asc())
        else:
            # Relevance: default to newest
            q = q.order_by(Document.created_at.desc())

        # Pagination
        docs = q.offset((page - 1) * per_page).limit(per_page).all()

        # Score by relevance if query given
        results = []
        for doc in docs:
            doc_dict = doc.to_dict()
            if query:
                text = f"{doc.title} {doc.summary or ''} {doc.content[:500] if doc.content else ''}"
                score = cosine_similarity_text(query, text)
                doc_dict["relevance_score"] = round(score, 4)
            else:
                doc_dict["relevance_score"] = 1.0
            results.append(doc_dict)

        if query and sort_by == "relevance":
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "results": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
            "query": query,
            "filters": {
                "ipc_section": ipc_section,
                "judge_name": judge_name,
                "case_type": case_type,
                "act_name": act_name,
                "party_name": party_name,
                "court": court,
            },
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"results": [], "total": 0, "page": page, "per_page": per_page,
                "total_pages": 1, "error": str(e)}


def search_ipc_sections(section_query: str, top_k: int = 10) -> List[Dict]:
    """Search IPC sections from the legal_acts table."""
    try:
        from schema import LegalAct

        results = LegalAct.query.filter(
            LegalAct.section_number.ilike(f"%{section_query}%") |
            LegalAct.section_title.ilike(f"%{section_query}%") |
            LegalAct.section_content.ilike(f"%{section_query}%")
        ).limit(top_k).all()

        return [r.to_dict() for r in results]
    except Exception as e:
        logger.error(f"IPC search error: {e}")
        return []


def full_text_search(query: str, top_k: int = 10) -> List[Dict]:
    """Full-text search across all documents with relevance ranking."""
    try:
        from schema import Document

        docs = Document.query.filter(
            Document.content.ilike(f"%{query}%") |
            Document.title.ilike(f"%{query}%")
        ).limit(top_k * 3).all()

        scored = []
        for doc in docs:
            text = f"{doc.title} {doc.summary or ''} {(doc.content or '')[:1000]}"
            score = cosine_similarity_text(query, text)
            d = doc.to_dict()
            d["relevance_score"] = round(score, 4)
            scored.append(d)

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:top_k]
    except Exception as e:
        logger.error(f"Full text search error: {e}")
        return []


def boolean_search(query: str) -> List[Dict]:
    """
    Boolean query parsing: AND, OR, NOT operators.
    Example: "murder AND IPC 302 NOT bail"
    """
    try:
        from schema import Document
        from sqlalchemy import and_, or_, not_

        # Parse operators
        and_terms = []
        or_terms = []
        not_terms = []

        tokens = query.split()
        i = 0
        current_op = "AND"
        while i < len(tokens):
            t = tokens[i].upper()
            if t in ("AND", "OR", "NOT"):
                current_op = t
            else:
                if current_op == "AND":
                    and_terms.append(tokens[i])
                elif current_op == "OR":
                    or_terms.append(tokens[i])
                elif current_op == "NOT":
                    not_terms.append(tokens[i])
            i += 1

        q = Document.query

        for term in and_terms:
            q = q.filter(
                Document.content.ilike(f"%{term}%") |
                Document.title.ilike(f"%{term}%")
            )

        if or_terms:
            or_filters = []
            for term in or_terms:
                or_filters.append(Document.content.ilike(f"%{term}%"))
                or_filters.append(Document.title.ilike(f"%{term}%"))
            q = q.filter(or_(*or_filters))

        for term in not_terms:
            q = q.filter(~Document.content.ilike(f"%{term}%"))

        docs = q.limit(20).all()
        return [d.to_dict() for d in docs]
    except Exception as e:
        logger.error(f"Boolean search error: {e}")
        return []


def get_popular_searches(limit: int = 10) -> List[Dict]:
    """Return most frequent search queries."""
    try:
        from schema import SearchHistory
        from sqlalchemy import func

        results = (
            SearchHistory.query
            .with_entities(SearchHistory.query, func.count(SearchHistory.id).label("count"))
            .group_by(SearchHistory.query)
            .order_by(func.count(SearchHistory.id).desc())
            .limit(limit)
            .all()
        )
        return [{"query": r.query, "count": r.count} for r in results]
    except Exception as e:
        logger.error(f"Popular searches error: {e}")
        return []


def get_search_suggestions(prefix: str, limit: int = 5) -> List[str]:
    """Auto-complete search suggestions based on previous queries."""
    try:
        from schema import SearchHistory

        results = SearchHistory.query.filter(
            SearchHistory.query.ilike(f"{prefix}%")
        ).order_by(SearchHistory.created_at.desc()).limit(limit).all()

        return list(set(r.query for r in results))
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return []


def save_search_history(query: str, search_type: str, results_count: int,
                         user_id: int = None, filters: dict = None):
    """Save a search query to history."""
    try:
        from schema import SearchHistory
        from database import db
        import json

        sh = SearchHistory(
            query=query,
            search_type=search_type,
            results_count=results_count,
            filters_used=json.dumps(filters) if filters else None,
            user_id=user_id,
        )
        db.session.add(sh)
        db.session.commit()
    except Exception as e:
        logger.warning(f"Save search history failed: {e}")
