"""
database.py — Database initialization, session management, and seeding
=======================================================================
"""

import json
import csv
from datetime import datetime
from pathlib import Path

from flask import Flask
from flask_bcrypt import Bcrypt

from schema import db, User, LegalAct
from config import (
    DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS,
    ADMIN_EMAIL, ADMIN_PASSWORD,
    INDIAN_LAWS_CSV, IPC_SECTIONS_CSV
)
from utils import logger

bcrypt_instance = Bcrypt()


def init_db(app: Flask):
    """Initialize Flask-SQLAlchemy with the app."""
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    db.init_app(app)
    bcrypt_instance.init_app(app)

    with app.app_context():
        db.create_all()
        _seed_admin(app)
        _seed_legal_acts(app)
        logger.info("Database initialized successfully.")


def _seed_admin(app: Flask):
    """Create default admin user if it doesn't exist."""
    with app.app_context():
        existing = User.query.filter_by(email=ADMIN_EMAIL).first()
        if not existing:
            pw_hash = bcrypt_instance.generate_password_hash(ADMIN_PASSWORD).decode("utf-8")
            admin = User(
                username="admin",
                email=ADMIN_EMAIL,
                password_hash=pw_hash,
                role="admin",
                full_name="System Administrator",
                is_active=True,
                is_verified=True,
            )
            db.session.add(admin)

            # Seed demo users
            for role, uname, email, name in [
                ("lawyer", "lawyer1", "lawyer@legalai.in", "Adv. Ramesh Kumar"),
            ]:
                pw = bcrypt_instance.generate_password_hash("Pass@123").decode("utf-8")
                u = User(
                    username=uname, email=email, password_hash=pw,
                    role=role, full_name=name, is_active=True, is_verified=True
                )
                db.session.add(u)

            db.session.commit()
            logger.info(f"Admin user created: {ADMIN_EMAIL}")


def _seed_legal_acts(app: Flask):
    """Seed legal acts from CSVs if table is empty."""
    with app.app_context():
        if LegalAct.query.count() > 0:
            return

        rows_added = 0

        # Load IPC sections
        if Path(IPC_SECTIONS_CSV).exists():
            try:
                with open(IPC_SECTIONS_CSV, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        act = LegalAct(
                            act_name=row.get("act_name", "Indian Penal Code"),
                            act_year=int(row.get("year", 1860)),
                            section_number=row.get("section", ""),
                            section_title=row.get("title", ""),
                            section_content=row.get("description", ""),
                            category=row.get("category", "Criminal"),
                            punishment=row.get("punishment", ""),
                            bailable=row.get("bailable", "No").strip().lower() == "yes",
                            cognizable=row.get("cognizable", "Yes").strip().lower() == "yes",
                        )
                        db.session.add(act)
                        rows_added += 1
                db.session.commit()
                logger.info(f"Seeded {rows_added} IPC sections into legal_acts table.")
            except Exception as e:
                logger.warning(f"Failed to seed IPC sections: {e}")
                db.session.rollback()


def get_db_stats() -> dict:
    """Return counts of all major tables."""
    from schema import Document, Case, ChatHistory, Prediction, SearchHistory
    try:
        return {
            "users": User.query.count(),
            "documents": Document.query.count(),
            "cases": Case.query.count(),
            "chats": ChatHistory.query.count(),
            "predictions": Prediction.query.count(),
            "searches": SearchHistory.query.count(),
            "legal_acts": LegalAct.query.count(),
        }
    except Exception as e:
        logger.error(f"DB stats error: {e}")
        return {}


def verify_user(email: str, password: str) -> "User | None":
    """Verify user credentials and return User object."""
    user = User.query.filter_by(email=email).first()
    if user and bcrypt_instance.check_password_hash(user.password_hash, password):
        user.last_login = datetime.utcnow()
        db.session.commit()
        return user
    return None


def create_user(username: str, email: str, password: str, role: str = "user",
                full_name: str = "") -> "User | None":
    """Create a new user with hashed password."""
    try:
        if User.query.filter_by(email=email).first():
            return None
        if User.query.filter_by(username=username).first():
            return None
        pw_hash = bcrypt_instance.generate_password_hash(password).decode("utf-8")
        user = User(
            username=username,
            email=email,
            password_hash=pw_hash,
            role=role,
            full_name=full_name,
            is_active=True,
            is_verified=True,
        )
        db.session.add(user)
        db.session.commit()
        logger.info(f"New user created: {email} [{role}]")
        return user
    except Exception as e:
        logger.error(f"create_user error: {e}")
        db.session.rollback()
        return None
