# ⚖ AI Legal Assistant for Indian Law
### RAG + Ollama + NLP + ML + Dynamic Programming | Final Year Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3%2FMistral-green)](https://ollama.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 One-Line Run Commands

### ▶ macOS / Linux
```bash
pip3 install -r requirements.txt --break-system-packages && pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz --break-system-packages && python3 app.py

PORT=8082 python3 app.py

```

> If you're using a **virtual environment** (recommended):
> ```bash
> python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz && python app.py

set PORT=8082 && python app.py
> ```

### ▶ Windows
```cmd
pip install -r requirements.txt && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz && python app.py
```

> Open your browser at → **http://localhost:8080**

> **Note:** If port 8080 is busy (e.g. "Address already in use"), change it with `PORT=8082 python3 app.py` (macOS) or `set PORT=8082 && python app.py` (Windows).

---

## 📸 Features at a Glance

| Feature | Technology |
|---|---|
| 🤖 AI Chat with RAG | Ollama (Llama3/Mistral) + ChromaDB |
| 📄 Document Upload & OCR | PyMuPDF + python-docx + pytesseract |
| 🔍 Advanced Legal Search | BM25 + Semantic Search + IPC Lookup |
| 🧠 Case Type Prediction | TF-IDF + Logistic Regression + Random Forest |
| 🌐 Multi-language Support | deep-translator (Tamil, Hindi, Telugu, etc.) |
| 📊 Dynamic Programming | LCS + Edit Distance + Jaccard Similarity |
| 🔐 Auth System | Flask-Login + Bcrypt |
| 🕸 Knowledge Graph | D3.js interactive visualization |
| 📈 Dashboard Analytics | Chart.js (Pie + Bar charts) |
| 📤 Export | PDF + CSV export |

---

## 🏗 Architecture

```
ai_legal_assistant/
│
├── app.py                  # Flask main app — all routes
├── config.py               # Configuration, constants
├── schema.py               # SQLAlchemy models (User, Document, Case, ...)
├── database.py             # DB init, seeding, utilities
│
├── rag_engine.py           # RAG pipeline (retrieval + generation)
├── ollama_engine.py        # Ollama LLM integration
├── document_processor.py   # PDF/DOCX/TXT/OCR processing
├── embeddings.py           # SentenceTransformers embeddings
├── vector_store.py         # ChromaDB vector store
│
├── nlp_engine.py           # spaCy NLP, keyword extraction, NER
├── ml_engine.py            # TF-IDF, ML classifiers
├── dp_similarity.py        # LCS, Edit Distance (Dynamic Programming)
├── case_predictor.py       # Case type prediction
├── translator.py           # Multi-language translation
├── search_engine.py        # BM25 + hybrid search
│
├── utils.py                # Logging, helpers, export
├── models.py               # ML model file (sklearn joblib)
│
├── datasets/
│   ├── indian_laws.csv     # 80+ Indian Acts with details
│   ├── ipc_sections.csv    # 90+ IPC/IT Act/POCSO sections
│   └── case_types.csv      # 100 sample legal cases
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Base layout (sidebar, topbar)
│   ├── index.html          # Dashboard
│   ├── chat.html           # AI Chat interface
│   ├── upload.html         # Document upload
│   ├── search.html         # Search engine
│   ├── login.html          # Login page
│   ├── register.html       # Registration
│   ├── admin.html          # Admin panel
│   └── error.html          # Error pages
│
├── static/
│   └── style.css           # Premium dark/light CSS
│
├── uploaded_documents/     # Uploaded files (auto-created)
├── chroma_db/              # Vector embeddings (auto-created)
├── legal.db                # SQLite database (auto-created)
└── requirements.txt
```

---

## ⚙️ Setup Guide

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | Latest |
| Ollama | Latest (for AI chat) |
| Tesseract OCR | Optional (for image OCR) |

---

### Step 1 — Install Ollama (for AI Chat)

**macOS:**
```bash
brew install ollama
ollama serve &
ollama pull llama3
```

**Windows:**
Download from → https://ollama.ai/download → Install → Run:
```cmd
ollama serve
ollama pull llama3
```

> **Note:** The app works even WITHOUT Ollama (falls back to extractive summaries and template answers). Ollama only powers the generative chat.

---

### Step 2 — Install Python Dependencies

**macOS / Linux:**
```bash
pip3 install -r requirements.txt --break-system-packages
```

**Windows:**
```cmd
pip install -r requirements.txt
```

If you encounter issues on macOS, install package-by-package:
```bash
pip3 install flask flask-sqlalchemy flask-login flask-bcrypt --break-system-packages
pip3 install chromadb sentence-transformers --break-system-packages
pip3 install spacy nltk pymupdf python-docx --break-system-packages
pip3 install scikit-learn pandas numpy joblib rank-bm25 --break-system-packages
pip3 install deep-translator langdetect requests pillow cachetools reportlab --break-system-packages
```

---

### Step 3 — (Optional) Install Tesseract for OCR

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from → https://github.com/UB-Mannheim/tesseract/wiki → Install to `C:\Program Files\Tesseract-OCR`

---

### Step 4 — Run the Application

**macOS / Linux (one line):**
```bash
pip3 install -r requirements.txt --break-system-packages && pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz --break-system-packages && python3 app.py
```

**macOS / Linux (with virtual environment — recommended):**
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz && python app.py
```

**Windows (one line):**
```cmd
pip install -r requirements.txt && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz && python app.py
```

Open → **http://localhost:8080**

> 💡 If port 8080 is busy: `PORT=8082 python3 app.py` (macOS) or `set PORT=8082 && python app.py` (Windows)

---

## 🔑 Default Login Credentials

| Role | Email | Password |
|---|---|---|
| **Admin** | admin@legalai.in | Admin@123 |
| **Lawyer** | lawyer@legalai.in | Pass@123 |

> These are seeded automatically on first run via `database.py`.

---

## 🔬 Technical Details

### RAG Pipeline
```
User Query
    ↓
NLP Preprocessing (spaCy tokenization, stopword removal)
    ↓
Hybrid Retrieval (ChromaDB semantic 60% + BM25 keyword 40%)
    ↓
Context Assembly (top-k chunks with metadata)
    ↓
Prompt Construction (system + legal context + conversation history)
    ↓
Ollama LLM (Llama3 / Mistral) → Answer
    ↓
Hallucination Detection (DP similarity score)
    ↓
Response to User (with confidence score + source citations)
```

### ML Pipeline
```
Legal Text Input
    ↓
TF-IDF Vectorization (10000 features, bigrams)
    ↓
Ensemble: LogisticRegression + RandomForest + MultinomialNB
    ↓
Voting Classifier (soft voting)
    ↓
Case Type Prediction + Confidence + Probabilities
```

### Dynamic Programming Algorithms
- **LCS (Longest Common Subsequence)**: Document similarity with memoization
- **Edit Distance (Levenshtein)**: Fuzzy matching for legal terms
- **Jaccard Similarity**: Set-based overlap for case matching
- **DP Subsequence**: Legal clause comparison

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET/POST | `/chat` | AI Chat UI |
| POST | `/api/chat` | Chat API (JSON) |
| GET/POST | `/upload` | Upload documents |
| GET | `/search` | Search interface |
| POST | `/api/predict` | Case prediction API |
| POST | `/summary` | Generate summary |
| POST | `/translate` | Translate text |
| GET | `/api/status` | System health |
| GET | `/api/ipc/<section>` | IPC section lookup |
| GET | `/api/knowledge-graph` | Graph data |
| POST | `/api/similarity` | DP similarity check |
| GET | `/admin` | Admin panel |

---

## 📋 Dataset Information

### `datasets/ipc_sections.csv`
- **90+ rows** covering IPC, IT Act 2000, Consumer Protection Act, POCSO, DV Act, NDPS, Motor Vehicles Act, and more
- Fields: `act_name, year, section, title, description, category, punishment, bailable, cognizable`

### `datasets/indian_laws.csv`
- **80+ rows** covering all major Indian Acts
- Fields: `act_name, year, category, description, court, key_provisions`

### `datasets/case_types.csv`
- **100 sample cases** with real-world legal scenarios
- Fields: `case_id, title, description, category, ipc_sections, court, year, state, verdict, petitioner, respondent, judge`

---

## 🎨 UI Features

- **Dark / Light theme** toggle (persisted in localStorage)
- **Glassmorphism** design with animated background shapes
- **Sidebar navigation** with collapsible mode
- **Real-time typing indicator** in AI chat
- **Confidence score badges** (High / Medium / Low)
- **Hallucination warning** for low-confidence answers
- **Source citation chips** linking to source documents
- **Follow-up question suggestions** after each AI answer
- **Drag-and-drop** file upload with animated progress
- **Voice input** (Web Speech API)
- **D3.js Knowledge Graph** visualization
- **Chart.js** analytics dashboard
- **Markdown rendering** for AI responses

---

## 📦 Key Dependencies

```
flask==3.0.3
flask-sqlalchemy==3.1.1
flask-login==0.6.3
flask-bcrypt==1.0.1
chromadb==0.5.3
sentence-transformers==3.0.1
langchain==0.2.16
langchain-community==0.2.16
spacy==3.7.6
nltk==3.8.1
pymupdf==1.24.9
python-docx==1.1.2
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
deep-translator==1.11.4
requests==2.32.3
pillow==10.4.0
pytesseract==0.3.13
joblib==1.4.2
rank-bm25==0.2.2
reportlab==4.2.2
```

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: spacy` | Run: `pip install spacy && python -m spacy download en_core_web_sm` |
| `Ollama connection refused` | Run: `ollama serve` in a separate terminal |
| `ChromaDB permission error` | Delete `chroma_db/` folder and restart |
| `OCR not working` | Install Tesseract and set path in `config.py` |
| `Database error` | Delete `legal.db` and restart (auto-reseeds) |
| `Port 5000 in use` | Change `PORT = 5001` in `config.py` |
| `en_core_web_sm not found` | Run: `python -m spacy download en_core_web_sm` |

---

## 📝 Academic Details

| Item | Details |
|---|---|
| **Project Type** | Final Year Engineering Project |
| **Domain** | Legal Technology (LegalTech) |
| **Core Technologies** | RAG, NLP, ML, Dynamic Programming |
| **AI Models** | Llama3 / Mistral (via Ollama, fully offline) |
| **Database** | SQLite (via SQLAlchemy ORM) |
| **Vector DB** | ChromaDB (local persistence) |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Frontend** | Jinja2 + Vanilla CSS + Chart.js + D3.js |
| **Backend** | Python 3.10+ + Flask 3.x |

---

## 👨‍💻 Project Author

Built as a final-year B.E./B.Tech Computer Science / AI project.

> **Legal Disclaimer:** This AI assistant provides legal information for educational purposes only. Always consult a qualified advocate for actual legal advice.

---

*⚖ LegalAI India — Making Indian Law accessible to everyone through AI*
