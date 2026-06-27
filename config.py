"""
config.py — Central Configuration for AI Legal Assistant
=========================================================
All settings, paths, model names, and feature flags in one place.
"""

import os
from pathlib import Path

# ── Base Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
for d in [UPLOAD_DIR, CHROMA_DIR, MODELS_DIR, DATASETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Flask Config ───────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "ai-legal-assistant-secret-2024-india")
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URI = f"sqlite:///{BASE_DIR / 'legal.db'}"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = False

# ── Ollama Config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_DEFAULT_MODEL = "llama3"
OLLAMA_AVAILABLE_MODELS = ["llama3", "mistral", "gemma3", "llama2"]
OLLAMA_TIMEOUT = 30
OLLAMA_MAX_TOKENS = 2048
OLLAMA_TEMPERATURE = 0.3

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 32

# ── ChromaDB Config ────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "legal_documents"
CHROMA_DISTANCE_METRIC = "cosine"
RETRIEVAL_TOP_K = 5
RETRIEVAL_HYBRID_ALPHA = 0.7  # 0=keyword, 1=semantic

# ── Document Processing ────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "png", "jpg", "jpeg", "tiff"}
MIN_CHUNK_LENGTH = 50

# ── NLP Config ────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"
MAX_SUMMARY_SENTENCES = 5
MAX_KEYWORDS = 15
NLP_BATCH_SIZE = 100

# ── ML / Case Classifier ──────────────────────────────────────────────────────
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

CASE_CATEGORIES = [
    "Criminal", "Civil", "Family", "Cyber Crime",
    "Property", "Labour", "Consumer", "Traffic"
]

ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 3)

# ── Dynamic Programming ───────────────────────────────────────────────────────
DP_CACHE_SIZE = 1024
SIMILARITY_THRESHOLD = 0.6

# ── Translation ───────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
}
DEFAULT_LANGUAGE = "en"

# ── Auth & Security ───────────────────────────────────────────────────────────
BCRYPT_ROUNDS = 12
SESSION_PERMANENT = False
USER_ROLES = ["admin", "lawyer", "student", "user"]
ADMIN_EMAIL = "admin@legalai.in"
ADMIN_PASSWORD = "Admin@123"  # Change in production

# ── Pagination ────────────────────────────────────────────────────────────────
ITEMS_PER_PAGE = 10
CHAT_HISTORY_LIMIT = 50

# ── Export ────────────────────────────────────────────────────────────────────
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "legal_assistant.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL = 3600  # seconds
CACHE_MAX_SIZE = 500

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "tesseract")
OCR_LANGUAGE = "eng+hin"

# ── Dataset Paths ─────────────────────────────────────────────────────────────
INDIAN_LAWS_CSV = DATASETS_DIR / "indian_laws.csv"
IPC_SECTIONS_CSV = DATASETS_DIR / "ipc_sections.csv"
CASE_TYPES_CSV = DATASETS_DIR / "case_types.csv"

# ── Feature Flags ─────────────────────────────────────────────────────────────
ENABLE_OCR = True
ENABLE_VOICE = True
ENABLE_TRANSLATION = True
ENABLE_KNOWLEDGE_GRAPH = True
ENABLE_TIMELINE = True
ENABLE_HALLUCINATION_CHECK = False
ENABLE_CITATION = True
