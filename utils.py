"""
utils.py — Helper utilities: logging, caching, export, text helpers
====================================================================
"""

import os
import re
import csv
import json
import time
import hashlib
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from functools import wraps
from cachetools import TTLCache, cached

from config import (
    LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    CACHE_TTL, CACHE_MAX_SIZE, EXPORT_DIR
)

# ── Logger Setup ──────────────────────────────────────────────────────────────
def setup_logger(name: str = "legal_assistant") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    # Rotating file handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        ))
        logger.addHandler(fh)
    except Exception:
        pass

    return logger


logger = setup_logger()

# ── Global Cache ──────────────────────────────────────────────────────────────
_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)


def cache_get(key: str):
    return _cache.get(key)


def cache_set(key: str, value):
    _cache[key] = value


def cache_delete(key: str):
    _cache.pop(key, None)


def cache_clear():
    _cache.clear()


def cache_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments."""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def memoize(func):
    """Simple memoization decorator using the global cache."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
        result = cache_get(key)
        if result is None:
            result = func(*args, **kwargs)
            cache_set(key, result)
        return result
    return wrapper


# ── Timing Decorator ──────────────────────────────────────────────────────────
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
        return result
    return wrapper


# ── Text Utilities ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove extra whitespace, control characters, and normalize."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0B80-\u0BFF]', ' ', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def sentence_split(text: str) -> list:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_year(text: str) -> list:
    return re.findall(r'\b(19|20)\d{2}\b', text)


def extract_dates(text: str) -> list:
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text, re.IGNORECASE))
    return list(set(dates))


def extract_ipc_sections(text: str) -> list:
    """Extract IPC section references from text."""
    patterns = [
        r'(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\(\d+\))?)',
        r'IPC\s+(\d+[A-Z]?)',
        r'u/s\s+(\d+[A-Z]?)',
        r'under\s+[Ss]ection\s+(\d+[A-Z]?)',
        r'BNS\s+(\d+[A-Z]?)',
    ]
    sections = []
    for p in patterns:
        sections.extend(re.findall(p, text))
    return list(set(sections))


def extract_case_numbers(text: str) -> list:
    patterns = [
        r'W\.?P\.?\s*\(C\)\s*No\.?\s*\d+/\d{4}',
        r'CRL\.?\s*A\.?\s*No\.?\s*\d+/\d{4}',
        r'Civil\s+Appeal\s+No\.?\s*\d+/\d{4}',
        r'SLP\s*\(Crl\)?\s*No\.?\s*\d+/\d{4}',
        r'\d+\s*of\s*\d{4}',
    ]
    numbers = []
    for p in patterns:
        numbers.extend(re.findall(p, text, re.IGNORECASE))
    return list(set(numbers))


def generate_hash(content: str) -> str:
    """SHA256 hash for duplicate detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def allowed_file(filename: str, allowed: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def safe_filename(filename: str) -> str:
    """Sanitize a filename."""
    filename = re.sub(r'[^\w\s\-.]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:200]


def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ── Export Utilities ──────────────────────────────────────────────────────────
def export_to_csv(data: list, filename: str, fieldnames: list = None) -> Path:
    """Export list of dicts to CSV."""
    path = EXPORT_DIR / filename
    if not data:
        return path
    fieldnames = fieldnames or list(data[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Exported CSV: {path}")
    return path


def export_to_json(data, filename: str) -> Path:
    path = EXPORT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def export_chat_to_pdf(chat_history: list, filename: str = None) -> Path:
    """Export chat history to PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.units import cm

        filename = filename or f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = EXPORT_DIR / filename

        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                     textColor=colors.HexColor('#1a237e'))
        story.append(Paragraph("⚖ AI Legal Assistant — Chat Export", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%"))
        story.append(Spacer(1, 0.5*cm))

        q_style = ParagraphStyle('Question', parent=styles['Normal'],
                                 textColor=colors.HexColor('#0d47a1'),
                                 spaceAfter=4)
        a_style = ParagraphStyle('Answer', parent=styles['Normal'],
                                 textColor=colors.HexColor('#1b5e20'),
                                 spaceBefore=4, spaceAfter=12)

        for item in chat_history:
            story.append(Paragraph(f"<b>Q:</b> {item.get('question', '')}", q_style))
            story.append(Paragraph(f"<b>A:</b> {item.get('answer', '')}", a_style))
            story.append(HRFlowable(width="100%", color=colors.lightgrey))
            story.append(Spacer(1, 0.3*cm))

        doc.build(story)
        logger.info(f"PDF exported: {path}")
        return path
    except ImportError:
        logger.warning("ReportLab not installed, falling back to text export")
        return export_to_json(chat_history, filename.replace('.pdf', '.json'))


def generate_citation(doc_title: str, section: str = None, year: str = None, page: int = None) -> str:
    """Generate Indian legal citation format."""
    citation = doc_title
    if year:
        citation += f" ({year})"
    if section:
        citation += f", Section {section}"
    if page:
        citation += f", p. {page}"
    return citation


# ── Response Helpers ──────────────────────────────────────────────────────────
def api_success(data=None, message="Success", status=200):
    return {"success": True, "message": message, "data": data}, status


def api_error(message="Error", status=400, details=None):
    resp = {"success": False, "message": message}
    if details:
        resp["details"] = details
    return resp, status


# ── Timestamp ─────────────────────────────────────────────────────────────────
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
