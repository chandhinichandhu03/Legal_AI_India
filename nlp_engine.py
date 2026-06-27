"""
nlp_engine.py — NLP module using spaCy + NLTK
===============================================
Named Entity Recognition, keyword extraction, summarization,
section detection, entity extraction, sentiment analysis.
"""

import re
import json
import string
from collections import Counter
from typing import List, Dict, Optional, Tuple

from utils import logger, clean_text, extract_dates, extract_ipc_sections, extract_case_numbers

# Lazy-load heavy models
_nlp = None
_stopwords_en = None
_lemmatizer = None

LEGAL_PERSON_KEYWORDS = {"judge", "justice", "hon", "petitioner", "respondent", "accused",
                          "plaintiff", "defendant", "appellant", "advocate", "adv", "dr", "mr", "mrs", "ms"}

COURT_NAMES = {
    "Supreme Court of India", "High Court", "District Court", "Sessions Court",
    "Magistrate Court", "Family Court", "Consumer Court", "Labour Court",
    "Cyber Crime Cell", "National Consumer Disputes Redressal Commission",
    "National Green Tribunal"
}

def _load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded.")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            try:
                import spacy
                spacy.cli.download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"spaCy load failed: {e}")
                _nlp = None
    return _nlp


def _load_nltk():
    global _stopwords_en, _lemmatizer
    if _stopwords_en is None:
        try:
            import nltk
            for pkg in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            _stopwords_en = set(stopwords.words("english"))
            _lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"NLTK load failed: {e}")
            _stopwords_en = set()
            _lemmatizer = None
    return _stopwords_en, _lemmatizer


# ── Core NLP Functions ────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Basic word tokenization."""
    _load_nltk()
    try:
        import nltk
        return nltk.word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    sw, _ = _load_nltk()
    punct = set(string.punctuation)
    return [t for t in tokens if t not in sw and t not in punct and len(t) > 2]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    _, lemm = _load_nltk()
    if lemm:
        return [lemm.lemmatize(t) for t in tokens]
    return tokens


def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """Extract top keywords using TF-based frequency after NLP preprocessing."""
    try:
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_tokens(tokens)

        # Filter: only alpha, length > 3
        tokens = [t for t in tokens if t.isalpha() and len(t) > 3]

        freq = Counter(tokens)
        return [word for word, _ in freq.most_common(top_n)]
    except Exception as e:
        logger.error(f"extract_keywords error: {e}")
        return []


def extractive_summarize(text: str, num_sentences: int = 5) -> str:
    """Extractive summarization: score sentences by keyword overlap."""
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:500]

        keywords = set(extract_keywords(text, top_n=20))
        scores = []
        for sent in sentences:
            words = set(sent.lower().split())
            score = len(words & keywords)
            scores.append(score)

        # Sort by score, pick top N
        ranked = sorted(zip(scores, range(len(sentences))), reverse=True)
        top_indices = sorted([idx for _, idx in ranked[:num_sentences]])
        summary = " ".join(sentences[i] for i in top_indices)
        return summary or sentences[0]
    except Exception as e:
        logger.error(f"summarize error: {e}")
        return text[:500]


def run_ner(text: str) -> Dict[str, List[str]]:
    """Run Named Entity Recognition using spaCy."""
    nlp = _load_spacy()
    entities = {
        "PERSON": [], "ORG": [], "GPE": [], "DATE": [],
        "MONEY": [], "LAW": [], "FAC": [], "EVENT": []
    }

    if not nlp:
        return entities

    try:
        doc = nlp(text[:100000])  # Limit to avoid memory issues
        for ent in doc.ents:
            label = ent.label_
            if label in entities:
                val = ent.text.strip()
                if val not in entities[label]:
                    entities[label].append(val)
    except Exception as e:
        logger.error(f"NER error: {e}")

    return entities


def extract_judge_names(text: str) -> List[str]:
    """Extract judge names using NER + pattern matching."""
    judges = []

    # Pattern: "Justice X Y", "Hon'ble X Y", "J. X Y"
    patterns = [
        r'(?:Justice|JUSTICE|Hon\'ble|HON\'BLE)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'\bJ\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'(?:Bench of|Coram:?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
    ]
    for p in patterns:
        found = re.findall(p, text)
        judges.extend(found)

    # Also try NER
    entities = run_ner(text[:5000])
    for person in entities.get("PERSON", []):
        words = person.lower().split()
        if any(kw in words for kw in ["justice", "judge", "hon"]):
            judges.append(person)

    return list(set(j.strip() for j in judges if len(j) > 3))


def extract_court_names(text: str) -> List[str]:
    """Extract court names using pattern matching."""
    found = []
    for court in COURT_NAMES:
        if court.lower() in text.lower():
            found.append(court)

    # Also regex
    patterns = [
        r'(?:in the|before the|before)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:Court|Tribunal|Commission))',
        r'([A-Z][a-z]+\s+High\s+Court)',
    ]
    for p in patterns:
        matches = re.findall(p, text)
        found.extend(matches)

    return list(set(found))


def extract_parties(text: str) -> Dict[str, List[str]]:
    """Extract petitioner and respondent names."""
    result = {"petitioner": [], "respondent": [], "appellant": [], "defendant": []}

    patterns = {
        "petitioner": [
            r'(?:Petitioner|PETITIONER|Plaintiff|PLAINTIFF)[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:\n|vs?\.|\bv/s\b|versus)',
        ],
        "respondent": [
            r'(?:Respondent|RESPONDENT|Defendant|DEFENDANT)[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:\n|$)',
        ],
        "appellant": [
            r'(?:Appellant|APPELLANT)[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:\n|vs?\.)',
        ],
    }

    for role, pats in patterns.items():
        for p in pats:
            found = re.findall(p, text)
            result[role].extend([f.strip() for f in found if len(f.strip()) > 3])

    return result


def detect_language(text: str) -> str:
    """Detect language of text."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "en"


def analyze_sentiment(text: str) -> Dict:
    """Simple rule-based sentiment for legal text."""
    positive_words = {"upheld", "granted", "allowed", "acquitted", "favorable",
                      "relief", "compensated", "awarded", "accepted"}
    negative_words = {"dismissed", "rejected", "denied", "convicted", "sentenced",
                      "penalty", "fine", "imprisonment", "quashed"}

    words = set(text.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    if pos > neg:
        sentiment = "Positive (Favorable)"
    elif neg > pos:
        sentiment = "Negative (Unfavorable)"
    else:
        sentiment = "Neutral"

    return {"sentiment": sentiment, "positive_signals": pos, "negative_signals": neg}


def detect_sections(text: str) -> List[Dict]:
    """Detect legal section headings in document."""
    patterns = [
        r'(?:SECTION|Section)\s+(\d+[A-Z]?\.?\d*)[:\s]+(.{10,80})',
        r'(?:CHAPTER|Chapter)\s+([IVX\d]+)[:\s]+(.{10,80})',
        r'(?:PART|Part)\s+([A-Z\d]+)[:\s]+(.{10,80})',
    ]
    sections = []
    for p in patterns:
        for m in re.finditer(p, text):
            sections.append({"number": m.group(1), "title": m.group(2).strip()})
    return sections


def extract_verdict(text: str) -> Optional[str]:
    """Extract verdict/judgment from case text."""
    patterns = [
        r'(?:JUDGMENT|ORDER|HELD|RESULT)[:\s]*(.{50,500}?)(?:\n\n|\Z)',
        r'(?:appeal is|petition is|application is)\s+(allowed|dismissed|granted|rejected)[^\n]*',
        r'(?:convicted|acquitted|sentenced to)\s+[^\n]+',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0).strip()[:300]
    return None


def extract_compensation(text: str) -> Optional[str]:
    """Extract compensation/fine amounts."""
    patterns = [
        r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{2})?(?:\s*(?:lakh|crore|lakhs|crores))?)',
        r'fine of\s+(Rs\.?\s*[\d,]+)',
        r'compensation of\s+(Rs\.?\s*[\d,]+)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def extract_timeline(text: str) -> List[Dict]:
    """Extract chronological events from case text."""
    dates = extract_dates(text)
    timeline = []
    for date in dates:
        # Find sentence containing this date
        idx = text.find(date)
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(text), idx + 150)
            context = text[start:end].strip()
            timeline.append({"date": date, "event": context})
    # Sort by year
    def sort_key(item):
        years = re.findall(r'\d{4}', item['date'])
        return years[0] if years else "0"
    return sorted(timeline, key=sort_key)


def full_analysis(text: str) -> Dict:
    """Run full NLP analysis on a document."""
    text = clean_text(text)

    return {
        "keywords": extract_keywords(text),
        "summary": extractive_summarize(text),
        "entities": run_ner(text),
        "judges": extract_judge_names(text),
        "courts": extract_court_names(text),
        "parties": extract_parties(text),
        "ipc_sections": extract_ipc_sections(text),
        "case_numbers": extract_case_numbers(text),
        "dates": extract_dates(text),
        "sections": detect_sections(text),
        "verdict": extract_verdict(text),
        "compensation": extract_compensation(text),
        "timeline": extract_timeline(text),
        "sentiment": analyze_sentiment(text),
        "language": detect_language(text),
        "word_count": len(text.split()),
    }
