"""
app.py — Main Flask Application
================================
All routes for the AI Legal Assistant for Indian Law.
"""

import os
import json
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, session, send_file, flash, abort
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    current_user
)
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY, DEBUG, HOST, PORT,
    UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    OLLAMA_AVAILABLE_MODELS, OLLAMA_DEFAULT_MODEL,
    SUPPORTED_LANGUAGES, ITEMS_PER_PAGE, EXPORT_DIR,
    CASE_CATEGORIES, ENABLE_OCR, ENABLE_VOICE, ENABLE_TRANSLATION
)
from schema import db, User, Document, Case, ChatHistory, Prediction, SearchHistory, LegalAct
from database import init_db, get_db_stats, verify_user, create_user
from utils import logger, allowed_file, safe_filename, export_to_csv, export_chat_to_pdf

# ── App Factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Login manager
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"

# Initialize DB
init_db(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Context Processors ────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    return {
        "app_name": "⚖ LegalAI India",
        "supported_languages": SUPPORTED_LANGUAGES,
        "case_categories": CASE_CATEGORIES,
        "current_year": datetime.now().year,
        "ollama_models": OLLAMA_AVAILABLE_MODELS,
        "enable_voice": ENABLE_VOICE,
        "enable_ocr": ENABLE_OCR,
        "enable_translation": ENABLE_TRANSLATION,
    }


# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = verify_user(email, password)
        if user:
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.full_name or user.username}! ⚖️", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        full_name = request.form.get("full_name", "").strip()
        role = request.form.get("role", "user")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif role not in ["user", "lawyer", "student"]:
            role = "user"
            user = create_user(username, email, password, role, full_name)
            if user:
                login_user(user)
                flash("Account created successfully! ⚖️", "success")
                return redirect(url_for("index"))
            else:
                flash("Username or email already exists.", "danger")
        else:
            user = create_user(username, email, password, role, full_name)
            if user:
                login_user(user)
                flash("Account created successfully! Welcome to LegalAI India ⚖️", "success")
                return redirect(url_for("index"))
            else:
                flash("Username or email already exists.", "danger")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    stats = get_db_stats()

    # Recent documents
    recent_docs = Document.query.order_by(Document.created_at.desc()).limit(5).all()

    # Recent chats
    recent_chats = ChatHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatHistory.created_at.desc()).limit(5).all()

    # Case type distribution
    from sqlalchemy import func
    case_dist = db.session.query(
        Document.case_type, func.count(Document.id)
    ).filter(Document.case_type.isnot(None)).group_by(Document.case_type).all()

    # Monthly activity (last 6 months)
    monthly_data = []
    for i in range(5, -1, -1):
        month_start = datetime.now().replace(day=1) - timedelta(days=30 * i)
        month_end = month_start + timedelta(days=30)
        count = Document.query.filter(
            Document.created_at >= month_start,
            Document.created_at < month_end
        ).count()
        monthly_data.append({
            "month": month_start.strftime("%b %Y"),
            "count": count
        })

    # Ollama status
    try:
        from ollama_engine import check_ollama_status
        ollama_status = check_ollama_status()
    except Exception:
        ollama_status = {"online": False, "models": []}

    # Popular IPC sections
    popular_sections = []
    try:
        from search_engine import get_popular_searches
        popular_sections = get_popular_searches(limit=5)
    except Exception:
        pass

    return render_template("index.html",
                           stats=stats,
                           recent_docs=[d.to_dict() for d in recent_docs],
                           recent_chats=[c.to_dict() for c in recent_chats],
                           case_distribution=dict(case_dist),
                           monthly_data=monthly_data,
                           ollama_status=ollama_status,
                           popular_searches=popular_sections)


# ── Upload Routes ─────────────────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({"success": False, "error": "File type not supported"}), 400

        try:
            from document_processor import save_uploaded_file, process_document
            from vector_store import store_embeddings, delete_vectors
            from nlp_engine import full_analysis
            from case_predictor import predict_case_type
            from ollama_engine import generate_summary

            # Save file
            file_path, unique_name = save_uploaded_file(file, file.filename)
            title = request.form.get("title", file.filename)

            # Process document
            proc_result = process_document(file_path, unique_name)

            if not proc_result.get("success"):
                return jsonify({"success": False, "error": proc_result.get("error", "Processing failed")}), 400

            raw_text = proc_result["raw_text"]
            content_hash = proc_result["content_hash"]

            # Duplicate check
            existing = Document.query.filter_by(hash_value=content_hash).first()
            if existing:
                return jsonify({
                    "success": False,
                    "error": f"Duplicate document detected: '{existing.title}' already exists.",
                    "existing_id": existing.id
                }), 409

            # NLP analysis
            analysis = full_analysis(raw_text)

            # Case prediction
            pred_result = predict_case_type(raw_text[:2000])
            case_type = pred_result.get("predicted_type", "Unknown")

            # AI Summary (if Ollama available)
            ai_summary = generate_summary(raw_text)
            summary = ai_summary if ai_summary else analysis.get("summary", "")

            # Create document record
            doc = Document(
                title=title,
                filename=unique_name,
                file_path=file_path,
                file_type=proc_result["file_type"],
                file_size=proc_result["file_size"],
                content=raw_text[:50000],
                summary=summary,
                keywords=json.dumps(analysis.get("keywords", [])),
                case_type=case_type,
                ipc_sections=json.dumps(analysis.get("ipc_sections", [])),
                case_numbers=json.dumps(analysis.get("case_numbers", [])),
                judges=json.dumps(analysis.get("judges", [])),
                parties=json.dumps(analysis.get("parties", {})),
                courts=json.dumps(analysis.get("courts", [])),
                dates_mentioned=json.dumps(analysis.get("dates", [])),
                language=analysis.get("language", "en"),
                word_count=proc_result["word_count"],
                chunk_count=proc_result["chunk_count"],
                hash_value=content_hash,
                is_indexed=False,
                ocr_processed=proc_result.get("ocr_processed", False),
                owner_id=current_user.id,
            )
            db.session.add(doc)
            db.session.flush()  # Get ID without full commit

            # Store embeddings
            chunks = proc_result["chunks"]
            chunk_metas = [{"doc_id": doc.id, "doc_title": title, "case_type": case_type}
                           for _ in chunks]
            stored = store_embeddings(chunks, doc.id, title, metadatas=chunk_metas)
            doc.is_indexed = stored > 0
            doc.chunk_count = stored

            # Also create prediction record
            pred_record = Prediction(
                input_text=raw_text[:500],
                predicted_type=case_type,
                confidence=pred_result.get("confidence", 0.0),
                all_probabilities=json.dumps(pred_result.get("all_probabilities", {})),
                model_name=pred_result.get("model_name", ""),
                document_id=doc.id,
                user_id=current_user.id,
            )
            db.session.add(pred_record)
            db.session.commit()

            logger.info(f"Document uploaded: {title} (ID={doc.id})")

            return jsonify({
                "success": True,
                "message": f"Document '{title}' uploaded and indexed successfully!",
                "doc_id": doc.id,
                "case_type": case_type,
                "chunks_stored": stored,
                "word_count": proc_result["word_count"],
                "summary": summary[:300],
                "keywords": analysis.get("keywords", [])[:10],
                "ipc_sections": analysis.get("ipc_sections", []),
                "judges": analysis.get("judges", []),
            })

        except Exception as e:
            db.session.rollback()
            logger.error(f"Upload error: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    # GET: render upload page
    user_docs = Document.query.filter_by(owner_id=current_user.id).order_by(
        Document.created_at.desc()
    ).limit(20).all()
    return render_template("upload.html", documents=[d.to_dict() for d in user_docs])


@app.route("/document/<int:doc_id>")
@login_required
def document_detail(doc_id):
    doc = Document.query.get_or_404(doc_id)
    try:
        from nlp_engine import extract_timeline
        timeline = extract_timeline(doc.content or "")
    except Exception:
        timeline = []
    return render_template("document_detail.html" if Path("templates/document_detail.html").exists()
                           else "upload.html", doc=doc.to_dict(), timeline=timeline)


@app.route("/document/<int:doc_id>/delete", methods=["POST"])
@login_required
def delete_document(doc_id):
    doc = Document.query.get_or_404(doc_id)

    # Only owner or admin
    if doc.owner_id != current_user.id and current_user.role != "admin":
        abort(403)

    try:
        from vector_store import delete_vectors
        delete_vectors(doc_id)

        # Delete file
        if doc.file_path and Path(doc.file_path).exists():
            Path(doc.file_path).unlink()

        db.session.delete(doc)
        db.session.commit()
        flash(f"Document '{doc.title}' deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting document: {e}", "danger")

    return redirect(url_for("upload"))


@app.route("/document/<int:doc_id>/favorite", methods=["POST"])
@login_required
def toggle_favorite(doc_id):
    doc = Document.query.get_or_404(doc_id)
    doc.is_favorite = not doc.is_favorite
    db.session.commit()
    return jsonify({"success": True, "is_favorite": doc.is_favorite})


# ── Chat Routes ───────────────────────────────────────────────────────────────
@app.route("/chat")
@login_required
def chat():
    session_id = session.get("chat_session_id")
    if not session_id:
        session["chat_session_id"] = str(uuid.uuid4())
        session_id = session["chat_session_id"]

    # Load history for current session
    history = ChatHistory.query.filter_by(
        session_id=session_id
    ).order_by(ChatHistory.created_at).limit(50).all()

    # Available models
    try:
        from ollama_engine import check_ollama_status
        ollama_status = check_ollama_status()
    except Exception:
        ollama_status = {"online": False, "models": []}

    return render_template("chat.html",
                           history=[h.to_dict() for h in history],
                           session_id=session_id,
                           ollama_status=ollama_status,
                           current_model=session.get("model", OLLAMA_DEFAULT_MODEL))


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """Main RAG chat endpoint."""
    data = request.get_json()
    question = (data.get("question") or "").strip()
    model = data.get("model", session.get("model", OLLAMA_DEFAULT_MODEL))
    language = data.get("language", "en")
    session_id = session.get("chat_session_id", str(uuid.uuid4()))

    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty"}), 400

    # Set model preference in session
    session["model"] = model
    session["chat_session_id"] = session_id

    try:
        # Build conversation history
        history_records = ChatHistory.query.filter_by(
            session_id=session_id
        ).order_by(ChatHistory.created_at.desc()).limit(10).all()

        conversation_history = []
        for h in reversed(history_records):
            conversation_history.append({"role": "user", "content": h.question})
            conversation_history.append({"role": "assistant", "content": h.answer})

        # RAG pipeline
        from rag_engine import answer_question
        result = answer_question(
            question=question,
            model=model,
            conversation_history=conversation_history,
            language=language,
        )

        # Save to DB
        chat_record = ChatHistory(
            session_id=session_id,
            question=question,
            answer=result.get("answer", ""),
            context_docs=json.dumps([s.get("doc_id") for s in result.get("sources", [])]),
            sources=json.dumps(result.get("sources", [])),
            model_used=result.get("model_used", model),
            language=language,
            confidence_score=result.get("confidence_score", 0.0),
            hallucination_score=result.get("hallucination_score", 0.0),
            hallucination_flag=result.get("hallucination_flag", False),
            response_time=result.get("response_time", 0.0),
            tokens_used=result.get("tokens_used", 0),
            citations=json.dumps(result.get("citations", [])),
            user_id=current_user.id,
        )
        db.session.add(chat_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "answer": result.get("answer", ""),
            "translated_answer": result.get("translated_answer"),
            "sources": result.get("sources", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "hallucination_flag": result.get("hallucination_flag", False),
            "hallucination_score": result.get("hallucination_score", 0.0),
            "citations": result.get("citations", []),
            "follow_up_questions": result.get("follow_up_questions", []),
            "response_time": result.get("response_time", 0.0),
            "model_used": result.get("model_used", model),
            "chat_id": chat_record.id,
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat/new-session", methods=["POST"])
@login_required
def new_chat_session():
    session["chat_session_id"] = str(uuid.uuid4())
    return jsonify({"success": True, "session_id": session["chat_session_id"]})


@app.route("/chat/feedback", methods=["POST"])
@login_required
def chat_feedback():
    data = request.get_json()
    chat_id = data.get("chat_id")
    helpful = data.get("helpful")
    if chat_id:
        chat = ChatHistory.query.get(chat_id)
        if chat:
            chat.is_helpful = helpful
            db.session.commit()
    return jsonify({"success": True})


# ── Search Routes ─────────────────────────────────────────────────────────────
@app.route("/search")
@login_required
def search():
    query = request.args.get("q", "")
    ipc_section = request.args.get("ipc", "")
    judge_name = request.args.get("judge", "")
    case_type = request.args.get("case_type", "")
    act_name = request.args.get("act", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    party_name = request.args.get("party", "")
    court = request.args.get("court", "")
    page = int(request.args.get("page", 1))
    sort_by = request.args.get("sort", "relevance")
    search_mode = request.args.get("mode", "standard")  # standard/boolean/ipc/fulltext

    results_data = {"results": [], "total": 0, "page": page,
                    "per_page": ITEMS_PER_PAGE, "total_pages": 1}

    if query or ipc_section or judge_name or case_type or date_from or date_to or party_name:
        try:
            from search_engine import (
                search_documents, search_ipc_sections,
                full_text_search, boolean_search, save_search_history
            )

            if search_mode == "ipc" and ipc_section:
                ipc_results = search_ipc_sections(ipc_section)
                results_data["ipc_results"] = ipc_results
            elif search_mode == "boolean" and query:
                results_data["results"] = boolean_search(query)
                results_data["total"] = len(results_data["results"])
            else:
                results_data = search_documents(
                    query=query,
                    ipc_section=ipc_section,
                    judge_name=judge_name,
                    case_type=case_type,
                    act_name=act_name,
                    date_from=date_from,
                    date_to=date_to,
                    party_name=party_name,
                    court=court,
                    page=page,
                    sort_by=sort_by,
                )

            # Save to search history
            save_search_history(
                query=query or ipc_section or judge_name or "",
                search_type=search_mode,
                results_count=results_data.get("total", 0),
                user_id=current_user.id,
                filters={"ipc": ipc_section, "judge": judge_name, "case_type": case_type}
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            flash(f"Search error: {e}", "danger")

    # IPC section quick reference
    ipc_quick = []
    if ipc_section:
        try:
            from search_engine import search_ipc_sections
            ipc_quick = search_ipc_sections(ipc_section, top_k=3)
        except Exception:
            pass

    return render_template("search.html",
                           results=results_data.get("results", []),
                           ipc_results=results_data.get("ipc_results", []),
                           total=results_data.get("total", 0),
                           page=page,
                           total_pages=results_data.get("total_pages", 1),
                           query=query,
                           ipc_quick=ipc_quick,
                           filters={
                               "ipc": ipc_section, "judge": judge_name,
                               "case_type": case_type, "act": act_name,
                               "date_from": date_from, "date_to": date_to,
                               "party": party_name, "court": court,
                               "sort": sort_by, "mode": search_mode,
                           })


@app.route("/api/search/suggestions")
@login_required
def search_suggestions():
    prefix = request.args.get("q", "")
    from search_engine import get_search_suggestions
    suggestions = get_search_suggestions(prefix)
    return jsonify({"suggestions": suggestions})


# ── Prediction Routes ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    similar_cases = []

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Please enter case description.", "warning")
        else:
            try:
                from case_predictor import predict_case_type, find_similar_cases_from_db, recommend_acts
                result = predict_case_type(text)

                # Find similar cases
                similar_cases = find_similar_cases_from_db(
                    text,
                    case_type=result.get("predicted_type"),
                    top_k=5
                )

                # Save prediction
                pred_record = Prediction(
                    input_text=text[:500],
                    predicted_type=result.get("predicted_type", "Unknown"),
                    confidence=result.get("confidence", 0.0),
                    all_probabilities=json.dumps(result.get("all_probabilities", {})),
                    model_name=result.get("model_name", ""),
                    similar_cases=json.dumps([s.get("title", "") for s in similar_cases[:3]]),
                    user_id=current_user.id,
                )
                db.session.add(pred_record)
                db.session.commit()

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash(f"Prediction error: {e}", "danger")

    return render_template("predict.html" if Path("templates/predict.html").exists() else "search.html",
                           result=result,
                           similar_cases=similar_cases)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Text required"}), 400

    try:
        from case_predictor import predict_case_type, find_similar_cases_from_db
        result = predict_case_type(text)
        similar = find_similar_cases_from_db(text, top_k=3)

        pred_record = Prediction(
            input_text=text[:500],
            predicted_type=result.get("predicted_type", "Unknown"),
            confidence=result.get("confidence", 0.0),
            all_probabilities=json.dumps(result.get("all_probabilities", {})),
            model_name=result.get("model_name", ""),
            user_id=current_user.id,
        )
        db.session.add(pred_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "result": result,
            "similar_cases": similar[:3],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Summary Route ─────────────────────────────────────────────────────────────
@app.route("/summary", methods=["POST"])
@login_required
def summary():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    doc_id = data.get("doc_id")

    if not text and doc_id:
        doc = Document.query.get(doc_id)
        if doc:
            text = doc.content or ""

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    try:
        from nlp_engine import extractive_summarize
        from ollama_engine import generate_summary

        extractive = extractive_summarize(text)
        ai_summary = generate_summary(text)

        if doc_id:
            doc = Document.query.get(doc_id)
            if doc and ai_summary:
                doc.summary = ai_summary
                db.session.commit()

        return jsonify({
            "success": True,
            "extractive_summary": extractive,
            "ai_summary": ai_summary,
            "word_count": len(text.split()),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Translation Route ─────────────────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
@login_required
def translate():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    target_lang = data.get("target_lang", "en")
    source_lang = data.get("source_lang", "auto")

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    try:
        from translator import translate_text
        result = translate_text(text, target_lang, source_lang)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── History Route ─────────────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    page = int(request.args.get("page", 1))
    session_filter = request.args.get("session", "")

    q = ChatHistory.query.filter_by(user_id=current_user.id)
    if session_filter:
        q = q.filter_by(session_id=session_filter)

    total = q.count()
    chats = q.order_by(ChatHistory.created_at.desc()).paginate(
        page=page, per_page=ITEMS_PER_PAGE, error_out=False
    )

    # Get distinct sessions
    sessions = db.session.query(ChatHistory.session_id).filter_by(
        user_id=current_user.id
    ).distinct().limit(20).all()

    return render_template("history.html" if Path("templates/history.html").exists() else "chat.html",
                           chats=chats,
                           total=total,
                           sessions=[s[0] for s in sessions],
                           current_session=session_filter)


# ── Export Routes ─────────────────────────────────────────────────────────────
@app.route("/export/chat/pdf")
@login_required
def export_chat_pdf():
    session_id = request.args.get("session", session.get("chat_session_id"))
    chats = ChatHistory.query.filter_by(
        session_id=session_id, user_id=current_user.id
    ).order_by(ChatHistory.created_at).all()

    chat_data = [{"question": c.question, "answer": c.answer} for c in chats]
    try:
        pdf_path = export_chat_to_pdf(chat_data)
        return send_file(str(pdf_path), as_attachment=True,
                         download_name=f"legal_chat_{datetime.now().strftime('%Y%m%d')}.pdf")
    except Exception as e:
        flash(f"PDF export failed: {e}", "danger")
        return redirect(url_for("history"))


@app.route("/export/documents/csv")
@login_required
def export_docs_csv():
    docs = Document.query.filter_by(owner_id=current_user.id).all()
    data = [d.to_dict() for d in docs]
    path = export_to_csv(data, f"documents_{datetime.now().strftime('%Y%m%d')}.csv")
    return send_file(str(path), as_attachment=True, download_name=path.name)


# ── Admin Panel ───────────────────────────────────────────────────────────────
@app.route("/admin")
@login_required
def admin():
    if current_user.role != "admin":
        abort(403)

    stats = get_db_stats()
    users = User.query.order_by(User.created_at.desc()).all()

    # Recent activity logs
    recent_docs = Document.query.order_by(Document.created_at.desc()).limit(10).all()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()

    # Vector store stats
    try:
        from vector_store import get_collection_stats
        vector_stats = get_collection_stats()
    except Exception:
        vector_stats = {}

    # ML model info
    try:
        from ml_engine import models_exist
        ml_ready = models_exist()
    except Exception:
        ml_ready = False

    return render_template("admin.html",
                           stats=stats,
                           users=[u.to_dict() for u in users],
                           recent_docs=[d.to_dict() for d in recent_docs],
                           recent_predictions=[p.to_dict() for p in recent_predictions],
                           vector_stats=vector_stats,
                           ml_ready=ml_ready)


@app.route("/admin/train-models", methods=["POST"])
@login_required
def admin_train_models():
    if current_user.role != "admin":
        abort(403)
    try:
        from ml_engine import train_models
        result = train_models()
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/user/<int:user_id>/toggle", methods=["POST"])
@login_required
def toggle_user(user_id):
    if current_user.role != "admin":
        abort(403)
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({"success": False, "error": "Cannot deactivate yourself"}), 400
    user.is_active = not user.is_active
    db.session.commit()
    return jsonify({"success": True, "is_active": user.is_active})


# ── API: Document Preview ─────────────────────────────────────────────────────
@app.route("/api/document/<int:doc_id>/preview")
@login_required
def api_doc_preview(doc_id):
    doc = Document.query.get_or_404(doc_id)
    return jsonify({
        "title": doc.title,
        "summary": doc.summary,
        "case_type": doc.case_type,
        "content_preview": (doc.content or "")[:1000],
        "keywords": json.loads(doc.keywords) if doc.keywords else [],
        "ipc_sections": json.loads(doc.ipc_sections) if doc.ipc_sections else [],
        "judges": json.loads(doc.judges) if doc.judges else [],
    })


# ── API: System Status ────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    try:
        from ollama_engine import check_ollama_status
        from vector_store import get_collection_stats
        ollama = check_ollama_status()
        vec = get_collection_stats()
    except Exception:
        ollama = {"online": False}
        vec = {}

    return jsonify({
        "app": "AI Legal Assistant for Indian Law",
        "version": "2.0.0",
        "status": "running",
        "ollama": ollama,
        "vector_store": vec,
        "db_stats": get_db_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    })


# ── API: IPC Quick Lookup ────────────────────────────────────────────────────
@app.route("/api/ipc/<section>")
@login_required
def api_ipc_lookup(section):
    from search_engine import search_ipc_sections
    results = search_ipc_sections(section, top_k=5)
    return jsonify({"section": section, "results": results})


# ── Knowledge Graph API ───────────────────────────────────────────────────────
@app.route("/api/knowledge-graph")
@login_required
def knowledge_graph():
    """Return graph data for D3.js visualization."""
    docs = Document.query.limit(50).all()
    nodes = []
    links = []

    for doc in docs:
        nodes.append({
            "id": f"doc_{doc.id}",
            "label": doc.title[:30],
            "type": "document",
            "case_type": doc.case_type or "Unknown",
        })

        # IPC section nodes
        sections = json.loads(doc.ipc_sections) if doc.ipc_sections else []
        for sec in sections[:3]:
            node_id = f"ipc_{sec}"
            if not any(n["id"] == node_id for n in nodes):
                nodes.append({"id": node_id, "label": f"IPC {sec}", "type": "section"})
            links.append({"source": f"doc_{doc.id}", "target": node_id, "type": "mentions"})

        # Judge nodes
        judges = json.loads(doc.judges) if doc.judges else []
        for j in judges[:2]:
            node_id = f"judge_{j[:20].replace(' ', '_')}"
            if not any(n["id"] == node_id for n in nodes):
                nodes.append({"id": node_id, "label": j[:20], "type": "judge"})
            links.append({"source": f"doc_{doc.id}", "target": node_id, "type": "presided_by"})

    return jsonify({"nodes": nodes, "links": links})


# ── DP Similarity API ─────────────────────────────────────────────────────────
@app.route("/api/similarity", methods=["POST"])
@login_required
def api_similarity():
    data = request.get_json()
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")

    if not text1 or not text2:
        return jsonify({"success": False, "error": "Both texts required"}), 400

    from dp_similarity import document_similarity_score
    result = document_similarity_score(text1, text2)
    return jsonify({"success": True, **result})


# ── Error Handlers ────────────────────────────────────────────────────────────
@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="Access Forbidden"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page Not Found"), 404


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    return render_template("error.html", code=500, message="Internal Server Error"), 500


# ── Template filter ───────────────────────────────────────────────────────────
@app.template_filter("format_date")
def format_date(dt_str):
    try:
        dt = datetime.strptime(str(dt_str)[:19], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return str(dt_str)


@app.template_filter("truncate_smart")
def truncate_smart(text, length=100):
    if not text:
        return ""
    text = str(text)
    if len(text) <= length:
        return text
    return text[:length].rsplit(' ', 1)[0] + "..."


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚖  AI Legal Assistant for Indian Law Starting...")
    logger.info(f"   Host: {HOST}:{PORT}")
    logger.info(f"   Debug: {DEBUG}")
    logger.info("=" * 60)

    # Pre-train ML models if not exist
    try:
        from ml_engine import models_exist, train_models
        if not models_exist():
            logger.info("Training ML models for first time...")
            train_models()
    except Exception as e:
        logger.warning(f"ML pre-training skipped: {e}")

    app.run(host=HOST, port=PORT, debug=DEBUG)
