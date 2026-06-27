"""
translator.py — Multilingual Translation Module
================================================
Translate legal text using deep-translator.
Supports English, Tamil, Hindi, Telugu, Kannada, Malayalam, Marathi, Bengali.
"""

import re
from typing import Optional, Dict, List
from functools import lru_cache

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from utils import logger


# ── Language Detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language of input text."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text[:500])
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


# ── Core Translation ──────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _translate_cached(text: str, src: str, dest: str) -> str:
    """Cached translation using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=src, target=dest)

        # Split long text into chunks (deep-translator limit ~5000 chars)
        if len(text) <= 4000:
            return translator.translate(text)
        else:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) < 4000:
                    current += " " + sent
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent
            if current:
                chunks.append(current.strip())

            translated_chunks = []
            for chunk in chunks:
                try:
                    t = GoogleTranslator(source=src, target=dest)
                    translated_chunks.append(t.translate(chunk))
                except Exception:
                    translated_chunks.append(chunk)
            return " ".join(translated_chunks)

    except ImportError:
        logger.warning("deep-translator not installed. pip install deep-translator")
        return text
    except Exception as e:
        logger.error(f"Translation error ({src}→{dest}): {e}")
        return text


def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> Dict:
    """
    Translate text to target language.
    Returns dict with translated text, source language, and metadata.
    """
    if not text or not text.strip():
        return {"translated": "", "source_lang": source_lang, "target_lang": target_lang, "success": False}

    if source_lang == "auto":
        source_lang = detect_language(text)

    if source_lang == target_lang:
        return {
            "translated": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "success": True,
            "note": "No translation needed",
        }

    # Validate languages
    if target_lang not in SUPPORTED_LANGUAGES:
        return {"translated": text, "error": f"Unsupported target language: {target_lang}", "success": False}

    translated = _translate_cached(text, source_lang, target_lang)

    return {
        "translated": translated,
        "original": text,
        "source_lang": source_lang,
        "source_lang_name": SUPPORTED_LANGUAGES.get(source_lang, source_lang),
        "target_lang": target_lang,
        "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
        "char_count": len(text),
        "success": True,
    }


def translate_answer(answer: str, question: str, target_lang: str) -> Dict:
    """Translate an AI-generated legal answer to target language."""
    if target_lang == "en" or not target_lang:
        return {"translated": answer, "target_lang": "en", "success": True}

    result = translate_text(answer, target_lang=target_lang)

    # Also translate the question for display
    q_result = translate_text(question, target_lang=target_lang)

    return {
        "translated_answer": result.get("translated", answer),
        "translated_question": q_result.get("translated", question),
        "target_lang": target_lang,
        "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
        "success": result.get("success", False),
    }


def translate_document_summary(summary: str, target_lang: str) -> str:
    """Translate document summary to target language."""
    result = translate_text(summary, target_lang=target_lang)
    return result.get("translated", summary)


def get_supported_languages() -> Dict[str, str]:
    """Return all supported languages."""
    return SUPPORTED_LANGUAGES


def get_language_name(code: str) -> str:
    """Get full name of a language code."""
    return SUPPORTED_LANGUAGES.get(code, code)


# ── Legal Term Glossary ───────────────────────────────────────────────────────

LEGAL_GLOSSARY = {
    "ta": {
        "petitioner": "மனுதாரர்",
        "respondent": "பதிலளிப்பவர்",
        "plaintiff": "வாதி",
        "defendant": "பிரதிவாதி",
        "judgment": "தீர்ப்பு",
        "verdict": "தீர்மானம்",
        "appeal": "மேல்முறையீடு",
        "bail": "ஜாமீன்",
        "advocate": "வழக்கறிஞர்",
        "court": "நீதிமன்றம்",
        "judge": "நீதிபதி",
        "FIR": "முதல் தகவல் அறிக்கை",
        "section": "பிரிவு",
    },
    "hi": {
        "petitioner": "याचिकाकर्ता",
        "respondent": "प्रतिवादी",
        "plaintiff": "वादी",
        "defendant": "प्रतिवादी",
        "judgment": "निर्णय",
        "verdict": "फैसला",
        "appeal": "अपील",
        "bail": "जमानत",
        "advocate": "अधिवक्ता",
        "court": "न्यायालय",
        "judge": "न्यायाधीश",
        "FIR": "प्रथम सूचना रिपोर्ट",
        "section": "धारा",
    },
}


def get_legal_term(term: str, lang: str) -> Optional[str]:
    """Get translation of a legal term in target language."""
    glossary = LEGAL_GLOSSARY.get(lang, {})
    return glossary.get(term.lower())


def enrich_with_glossary(text: str, lang: str) -> str:
    """Replace common legal terms with translated versions in output."""
    glossary = LEGAL_GLOSSARY.get(lang, {})
    for en_term, native_term in glossary.items():
        # Add native term in parentheses
        text = re.sub(
            r'\b' + re.escape(en_term) + r'\b',
            f"{en_term} ({native_term})",
            text,
            flags=re.IGNORECASE,
        )
    return text
