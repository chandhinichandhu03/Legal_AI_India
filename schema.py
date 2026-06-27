"""
schema.py — SQLAlchemy 2.0 ORM Models (Mapped / mapped_column style)
======================================================================
Using SQLAlchemy 2.0 declarative style so Pylance resolves __init__
keyword arguments correctly.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import ForeignKey, Text, Float, Integer, String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


# ── User ──────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    """Authenticated user with role-based access."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    profile_pic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship("Document", backref="owner", lazy="dynamic")
    chats: Mapped[list["ChatHistory"]] = relationship("ChatHistory", backref="user", lazy="dynamic")
    searches: Mapped[list["SearchHistory"]] = relationship("SearchHistory", backref="user", lazy="dynamic")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", backref="user", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<User {self.username} [{self.role}]>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "created_at": str(self.created_at),
            "last_login": str(self.last_login) if self.last_login else None,
        }


# ── Document ──────────────────────────────────────────────────────────────────

class Document(db.Model):
    """Uploaded legal document with extracted metadata."""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)       # JSON list
    case_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ipc_sections: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON list
    case_numbers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON list
    judges: Mapped[Optional[str]] = mapped_column(Text, nullable=True)         # JSON list
    parties: Mapped[Optional[str]] = mapped_column(Text, nullable=True)        # JSON list
    courts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)         # JSON list
    dates_mentioned: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    language: Mapped[str] = mapped_column(String(10), default="en")
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    hash_value: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    is_bookmarked: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    owner_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "summary": self.summary or "",
            "keywords": json.loads(self.keywords) if self.keywords else [],
            "case_type": self.case_type or "Unknown",
            "ipc_sections": json.loads(self.ipc_sections) if self.ipc_sections else [],
            "judges": json.loads(self.judges) if self.judges else [],
            "is_indexed": self.is_indexed,
            "is_favorite": self.is_favorite,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "created_at": str(self.created_at),
        }


# ── Case ──────────────────────────────────────────────────────────────────────

class Case(db.Model):
    """Legal case record extracted from documents."""
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    case_title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    case_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    court: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    judge_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    petitioner: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    respondent: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    ipc_sections: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON list
    act_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    compensation: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    date_filed: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    date_decided: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    facts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    issues: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    arguments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timeline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)       # JSON list
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)       # JSON list
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    document_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("documents.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[Optional["Document"]] = relationship("Document", backref="case")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_number": self.case_number,
            "case_title": self.case_title,
            "case_type": self.case_type,
            "court": self.court,
            "judge_name": self.judge_name,
            "petitioner": self.petitioner,
            "respondent": self.respondent,
            "ipc_sections": json.loads(self.ipc_sections) if self.ipc_sections else [],
            "verdict": self.verdict,
            "year": self.year,
            "state": self.state,
            "summary": self.summary,
            "created_at": str(self.created_at),
        }


# ── ChatHistory ───────────────────────────────────────────────────────────────

class ChatHistory(db.Model):
    """Conversation history for the RAG chatbot."""
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    context_docs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON list
    sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)        # JSON list
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="en")
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    hallucination_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    hallucination_score: Mapped[float] = mapped_column(Float, default=0.0)
    response_time: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    citations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # JSON list
    is_helpful: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "sources": json.loads(self.sources) if self.sources else [],
            "model_used": self.model_used,
            "language": self.language,
            "confidence_score": self.confidence_score,
            "hallucination_flag": self.hallucination_flag,
            "response_time": self.response_time,
            "citations": json.loads(self.citations) if self.citations else [],
            "created_at": str(self.created_at),
        }


# ── Prediction ────────────────────────────────────────────────────────────────

class Prediction(db.Model):
    """ML case type prediction records."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    all_probabilities: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON dict
    model_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    similar_cases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # JSON list
    document_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("documents.id"), nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "predicted_type": self.predicted_type,
            "confidence": round(self.confidence * 100, 2),
            "all_probabilities": json.loads(self.all_probabilities) if self.all_probabilities else {},
            "model_name": self.model_name,
            "created_at": str(self.created_at),
        }


# ── SearchHistory ─────────────────────────────────────────────────────────────

class SearchHistory(db.Model):
    """User search activity tracking."""
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    search_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    filters_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON dict
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "search_type": self.search_type,
            "results_count": self.results_count,
            "created_at": str(self.created_at),
        }


# ── LegalAct ──────────────────────────────────────────────────────────────────

class LegalAct(db.Model):
    """Indian legal acts and sections reference table."""
    __tablename__ = "legal_acts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    act_name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    act_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    section_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    punishment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bailable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    cognizable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "act_name": self.act_name,
            "act_year": self.act_year,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "section_content": self.section_content,
            "category": self.category,
        }
