"""
case_predictor.py — Legal Case Type Predictor with similar case retrieval
==========================================================================
"""

import json
from typing import Dict, List, Optional
from datetime import datetime

from ml_engine import predict_case_type_ml, get_feature_importance, models_exist, train_models
from dp_similarity import find_most_similar, document_similarity_score
from nlp_engine import extract_keywords, extractive_summarize
from utils import logger


CASE_DESCRIPTIONS = {
    "Criminal": {
        "description": "Cases involving violation of criminal law — murder, theft, rape, fraud, assault.",
        "common_acts": ["Indian Penal Code (IPC)", "CrPC", "NDPS Act", "Arms Act"],
        "typical_courts": ["Magistrate Court", "Sessions Court", "High Court", "Supreme Court"],
        "typical_outcome": "Imprisonment, fine, or acquittal",
        "color": "#ef4444",
        "icon": "⚖️",
    },
    "Civil": {
        "description": "Non-criminal disputes — contracts, property, torts, injunctions, recovery.",
        "common_acts": ["CPC", "Contract Act", "Transfer of Property Act", "Limitation Act"],
        "typical_courts": ["Civil Court", "District Court", "High Court"],
        "typical_outcome": "Decree, compensation, injunction",
        "color": "#3b82f6",
        "icon": "🏛️",
    },
    "Family": {
        "description": "Marriage, divorce, custody, maintenance, adoption, domestic violence.",
        "common_acts": ["Hindu Marriage Act", "CrPC Section 125", "DV Act", "Guardian Act"],
        "typical_courts": ["Family Court", "Magistrate Court"],
        "typical_outcome": "Divorce decree, custody order, maintenance",
        "color": "#ec4899",
        "icon": "👨‍👩‍👧",
    },
    "Cyber Crime": {
        "description": "Online offences — hacking, fraud, data breach, cyberbullying, phishing.",
        "common_acts": ["IT Act 2000", "IT Amendment Act 2008", "IPC Cyber provisions"],
        "typical_courts": ["Cyber Crime Cell", "Magistrate Court", "Sessions Court"],
        "typical_outcome": "Imprisonment, fine, account suspension",
        "color": "#8b5cf6",
        "icon": "💻",
    },
    "Property": {
        "description": "Land, real estate, tenancy, inheritance, acquisition, registration disputes.",
        "common_acts": ["Transfer of Property Act", "Registration Act", "LARR Act", "RERA"],
        "typical_courts": ["Civil Court", "Revenue Court", "High Court"],
        "typical_outcome": "Title deed, possession order, compensation",
        "color": "#f59e0b",
        "icon": "🏠",
    },
    "Labour": {
        "description": "Employment disputes — dismissal, wages, harassment, PF, ESI, union.",
        "common_acts": ["Industrial Disputes Act", "Payment of Wages Act", "ESI Act", "POSH Act"],
        "typical_courts": ["Labour Court", "Industrial Tribunal", "High Court"],
        "typical_outcome": "Reinstatement, compensation, back wages",
        "color": "#10b981",
        "icon": "👷",
    },
    "Consumer": {
        "description": "Consumer disputes — defective goods, service deficiency, insurance, banking.",
        "common_acts": ["Consumer Protection Act 2019", "Insurance Act", "Banking Regulation Act"],
        "typical_courts": ["District Consumer Forum", "State Commission", "NCDRC"],
        "typical_outcome": "Compensation, refund, product replacement",
        "color": "#06b6d4",
        "icon": "🛒",
    },
    "Traffic": {
        "description": "Road accidents, drunk driving, MACT claims, traffic violations.",
        "common_acts": ["Motor Vehicles Act", "MACT Rules", "Insurance Act"],
        "typical_courts": ["MACT Tribunal", "Magistrate Court", "High Court"],
        "typical_outcome": "Compensation, license suspension, fine",
        "color": "#f97316",
        "icon": "🚗",
    },
    "General / Non-Legal": {
        "description": "General text or non-legal document (e.g. resume, CV, personal profile, or generic writing).",
        "common_acts": ["None / Not Applicable"],
        "typical_courts": ["None / Not Applicable"],
        "typical_outcome": "No legal action required",
        "color": "#64748b",
        "icon": "📄",
    },
}


def predict_case_type(text: str, store_prediction: bool = True) -> Dict:
    """
    Full case type prediction pipeline:
    1. ML prediction with confidence
    2. Feature importance (XAI)
    3. Similar case retrieval
    4. Case category metadata
    """
    if not text or len(text.strip()) < 10:
        return {"error": "Input text too short for prediction."}

    # Ensure models exist
    if not models_exist():
        logger.info("Training ML models on first run...")
        try:
            train_models()
        except Exception as e:
            logger.error(f"Training failed: {e}")

    # ML prediction
    prediction = predict_case_type_ml(text)
    case_type = prediction.get("predicted_type", "Unknown")
    conf = prediction.get("confidence", 0.0)

    # Resume & non-legal document detection heuristics
    text_lower = text.lower()
    resume_keywords = ["resume", "cv", "curriculum vitae", "skills", "experience", "education", "employment history", "professional summary", "projects"]
    resume_matches = sum(1 for kw in resume_keywords if kw in text_lower)
    
    is_resume = resume_matches >= 3 or ("resume" in text_lower and resume_matches >= 2) or ("curriculum vitae" in text_lower)
    
    if is_resume or conf < 0.35:
        case_type = "General / Non-Legal"
        prediction["predicted_type"] = case_type
        # Set high confidence since we are certain this is a general/non-legal text
        prediction["confidence"] = 0.95 if is_resume else 0.88
        conf = prediction["confidence"]

    # Feature importance
    features = get_feature_importance(text)

    # Keywords from input text
    keywords = extract_keywords(text, top_n=10)
    summary = extractive_summarize(text, num_sentences=3)

    # Category metadata
    category_info = CASE_DESCRIPTIONS.get(case_type, {})

    # Confidence level label
    conf = prediction.get("confidence", 0.0)
    if conf >= 0.85:
        conf_label = "Very High"
    elif conf >= 0.70:
        conf_label = "High"
    elif conf >= 0.55:
        conf_label = "Moderate"
    elif conf >= 0.40:
        conf_label = "Low"
    else:
        conf_label = "Very Low"

    result = {
        "predicted_type": case_type,
        "confidence": round(conf, 4),
        "confidence_percent": round(conf * 100, 1),
        "confidence_label": conf_label,
        "all_probabilities": prediction.get("all_probabilities", {}),
        "model_name": prediction.get("model_name", "Unknown"),
        "top_features": features,
        "keywords": keywords,
        "summary": summary,
        "category_info": category_info,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result


def find_similar_cases_from_db(query_text: str, case_type: str = None, top_k: int = 5) -> List[Dict]:
    """
    Find similar cases from the database using DP similarity.
    """
    try:
        from schema import Document, Case
        from database import db

        q = Document.query
        if case_type:
            q = q.filter(Document.case_type == case_type)

        docs = q.limit(100).all()
        candidates = []
        for doc in docs:
            if doc.content:
                candidates.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content[:1000],
                    "case_type": doc.case_type,
                    "filename": doc.filename,
                    "created_at": str(doc.created_at),
                })

        if not candidates:
            return []

        return find_most_similar(query_text, candidates, top_k=top_k)
    except Exception as e:
        logger.error(f"find_similar_cases error: {e}")
        return []


def recommend_acts(case_type: str) -> List[str]:
    """Recommend relevant Indian acts based on case type."""
    info = CASE_DESCRIPTIONS.get(case_type, {})
    return info.get("common_acts", [])


def get_all_category_info() -> Dict:
    """Return metadata for all case categories."""
    return CASE_DESCRIPTIONS
