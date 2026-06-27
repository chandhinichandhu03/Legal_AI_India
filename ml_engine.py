"""
ml_engine.py — Machine Learning Engine for Legal Case Classification
=====================================================================
Train TF-IDF + Logistic Regression, Random Forest, Naive Bayes.
Compare models, save best model. Provides prediction with confidence.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from config import (
    CLASSIFIER_PATH, TFIDF_PATH, LABEL_ENCODER_PATH,
    CASE_CATEGORIES, ML_TEST_SIZE, ML_RANDOM_STATE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, DATASETS_DIR
)
from utils import logger


# ── Training Data ─────────────────────────────────────────────────────────────

TRAINING_SAMPLES = [
    # Criminal
    ("murder accused stabbed victim knife IPC 302 FIR filed police station Sessions Court", "Criminal"),
    ("theft robbery dacoity IPC 379 accused arrested stolen property recovered", "Criminal"),
    ("rape sexual assault victim FIR IPC 376 fast track court bail denied", "Criminal"),
    ("kidnapping ransom demand accused arrested police custody IPC 363 364", "Criminal"),
    ("assault grievous hurt IPC 326 accused hospital treatment victim police complaint", "Criminal"),
    ("cheating fraud IPC 420 complaint filed criminal court accused absconding", "Criminal"),
    ("forgery document IPC 465 accused convicted two years imprisonment", "Criminal"),
    ("drug trafficking NDPS Act narcotic substance accused arrested Customs Court", "Criminal"),
    ("extortion blackmail accused arrested police IPC 383 bail application", "Criminal"),
    ("dowry death IPC 304B husband mother-in-law arrested Sessions Court", "Criminal"),
    ("rioting unlawful assembly IPC 147 148 police lathi charge FIR registered", "Criminal"),
    ("attempt murder IPC 307 victim survived accused absconding warrant issued", "Criminal"),
    ("corruption bribe public servant Prevention of Corruption Act CBI investigation", "Criminal"),

    # Civil
    ("property dispute civil suit decree possession land registry ownership", "Civil"),
    ("breach contract damages civil court plaintiff defendant suit filed", "Civil"),
    ("injunction restraining order civil court temporary permanent relief", "Civil"),
    ("partition suit family property co-owner share decree court", "Civil"),
    ("defamation reputation damage civil suit compensation claimed court", "Civil"),
    ("specific performance contract enforcement civil court plaintiff defendant", "Civil"),
    ("money recovery suit principal interest decree execution court", "Civil"),
    ("easement rights passage property civil court neighboring land", "Civil"),
    ("mortgage foreclosure property bank loan default civil court", "Civil"),
    ("tort negligence damages civil court medical malpractice hospital", "Civil"),

    # Family
    ("divorce petition mutual consent grounds irretrievable breakdown marriage", "Family"),
    ("child custody guardianship welfare minor parent Family Court", "Family"),
    ("maintenance alimony wife children Section 125 CrPC Magistrate Court", "Family"),
    ("domestic violence Protection of Women Domestic Violence Act shelter home", "Family"),
    ("adoption minor child CARA registration guardianship petition", "Family"),
    ("matrimonial property division divorce settlement decree Family Court", "Family"),
    ("NRI marriage abandonment wife children maintenance decree", "Family"),
    ("marriage registration certificate Hindu Marriage Act parties", "Family"),
    ("dowry harassment IPC 498A husband in-laws FIR Family Court hearing", "Family"),
    ("conjugal rights restitution Section 9 Hindu Marriage Act petition decree", "Family"),

    # Cyber Crime
    ("hacking unauthorized access computer system IT Act Section 66 accused arrested", "Cyber Crime"),
    ("online fraud phishing email banking credentials stolen victim IT Act", "Cyber Crime"),
    ("cyberbullying harassment social media victim complaint police IT Act 67", "Cyber Crime"),
    ("data breach personal information stolen server IT Act CERT-In complaint", "Cyber Crime"),
    ("identity theft Aadhaar impersonation online fraud IT Act accused court", "Cyber Crime"),
    ("ransomware malware attack company system IT Act accused arrested", "Cyber Crime"),
    ("revenge porn obscene photo video shared online IT Act Section 67 victim", "Cyber Crime"),
    ("cryptocurrency fraud bitcoin scam online investment victim complaint police", "Cyber Crime"),
    ("ATM card cloning skimming banking fraud victim police complaint IT Act", "Cyber Crime"),
    ("fake news social media defamation IT Act complaint court accused", "Cyber Crime"),

    # Property
    ("sale deed registration stamp duty sub-registrar property transfer", "Property"),
    ("encroachment boundary dispute survey land Revenue Department", "Property"),
    ("tenant eviction landlord rent arrears premises tenancy law", "Property"),
    ("will probate executor estate distribution High Court Registry", "Property"),
    ("property tax arrears municipal corporation demand notice owner", "Property"),
    ("land acquisition compensation LARR Act collector objection court", "Property"),
    ("benami transaction property Act nominee ownership court", "Property"),
    ("builder developer delay possession flat buyer RERA complaint", "Property"),
    ("lease rent deed commercial property landlord tenant dispute", "Property"),
    ("mutation record revenue court patta chitta land ownership transfer", "Property"),

    # Labour
    ("wrongful termination dismissal employee Industrial Disputes Act Labour Court", "Labour"),
    ("unpaid wages salary arrears employer employee Labour Commissioner", "Labour"),
    ("PF ESI deduction non-remittance EPFO employer employee complaint", "Labour"),
    ("sexual harassment workplace POSH Act Internal Complaints Committee", "Labour"),
    ("workman compensation accident factory Employees Compensation Act", "Labour"),
    ("gratuity payment dispute employer Payment of Gratuity Act Labour Court", "Labour"),
    ("retrenchment layoff compensation Industrial Disputes Act worker", "Labour"),
    ("trade union recognition strike lockout dispute Industrial Relations Code", "Labour"),
    ("minimum wages violation employer Labour Inspector complaint penalty", "Labour"),
    ("contract labour principal employer Contractors Act worker dispute", "Labour"),

    # Consumer
    ("defective product consumer complaint NCDRC District Forum compensation", "Consumer"),
    ("insurance claim rejected company complaint State Consumer Commission", "Consumer"),
    ("bank service deficiency complaint consumer forum compensation awarded", "Consumer"),
    ("medical negligence hospital consumer forum compensation doctor", "Consumer"),
    ("airline flight delay cancellation compensation consumer forum complaint", "Consumer"),
    ("education institution fee refund admission cancellation consumer forum", "Consumer"),
    ("telecom service complaint TRAI consumer forum reimbursement", "Consumer"),
    ("e-commerce fraud product not delivered Amazon Flipkart consumer complaint", "Consumer"),
    ("real estate builder delay possession consumer forum RERA complaint", "Consumer"),
    ("electricity board billing error consumer forum compensation complaint", "Consumer"),

    # Traffic
    ("road accident vehicle collision FIR Motor Vehicles Act compensation claim", "Traffic"),
    ("drunk driving DUI arrested traffic police license cancelled", "Traffic"),
    ("MACT Motor Accident Claims Tribunal compensation family deceased accident", "Traffic"),
    ("hit run vehicle accident victim hospital MACT claim filed", "Traffic"),
    ("traffic signal violation challan fine Motor Vehicles Act penalty", "Traffic"),
    ("reckless driving dangerous overspeeding license suspended court", "Traffic"),
    ("invalid driving license vehicle seized court fine imposed traffic", "Traffic"),
    ("insurance claim third party accident Motor Vehicles Act insurer", "Traffic"),
    ("vehicle registration tax default RTO notice penalty owner", "Traffic"),
    ("pedestrian hit vehicle compensation MACT claim tribunal award", "Traffic"),
]


def _get_training_data() -> Tuple[List[str], List[str]]:
    """Get training data from built-in samples + CSV datasets."""
    texts = [t for t, _ in TRAINING_SAMPLES]
    labels = [l for _, l in TRAINING_SAMPLES]

    # Try loading from CSV dataset
    csv_path = Path(DATASETS_DIR) / "case_types.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "description" in df.columns and "category" in df.columns:
                texts.extend(df["description"].tolist())
                labels.extend(df["category"].tolist())
                logger.info(f"Loaded {len(df)} additional training samples from CSV.")
        except Exception as e:
            logger.warning(f"Failed to load CSV training data: {e}")

    return texts, labels


def train_models() -> Dict:
    """Train multiple classifiers and save the best one."""
    logger.info("Starting ML model training...")
    texts, labels = _get_training_data()

    # Filter valid categories
    valid = [(t, l) for t, l in zip(texts, labels) if l in CASE_CATEGORIES]
    if not valid:
        raise ValueError("No valid training data found.")
    texts, labels = zip(*valid)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=1,
    )
    X = tfidf.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )

    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=5.0, random_state=ML_RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=ML_RANDOM_STATE, n_jobs=-1
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
    }

    results = {}
    best_accuracy = 0.0
    best_model = None
    best_model_name = ""

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(clf, X, y, cv=min(5, len(set(labels))), scoring="accuracy")

            results[name] = {
                "accuracy": round(acc, 4),
                "cv_mean": round(cv_scores.mean(), 4),
                "cv_std": round(cv_scores.std(), 4),
                "report": classification_report(y_test, y_pred,
                                                target_names=le.classes_,
                                                output_dict=True),
            }

            logger.info(f"{name}: acc={acc:.4f}, cv={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = clf
                best_model_name = name
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")

    if best_model is None:
        raise RuntimeError("All classifiers failed to train.")

    # Save models
    joblib.dump(best_model, CLASSIFIER_PATH)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    logger.info(f"Best model: {best_model_name} (accuracy={best_accuracy:.4f})")
    logger.info(f"Models saved to {CLASSIFIER_PATH.parent}")

    return {
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "results": results,
        "classes": le.classes_.tolist(),
        "training_samples": len(texts),
    }


def load_models() -> Tuple:
    """Load trained models from disk."""
    try:
        clf = joblib.load(CLASSIFIER_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        return clf, tfidf, le
    except FileNotFoundError:
        logger.warning("Models not found. Training now...")
        train_models()
        clf = joblib.load(CLASSIFIER_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        return clf, tfidf, le


def predict_case_type_ml(text: str) -> Dict:
    """Predict case type with probability scores."""
    try:
        clf, tfidf, le = load_models()
        X = tfidf.transform([text])

        pred_idx = clf.predict(X)[0]
        predicted = le.inverse_transform([pred_idx])[0]

        # Get probabilities if available
        proba = {}
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            proba = {
                le.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probs)
            }
            confidence = float(max(probs))
        else:
            proba = {predicted: 1.0}
            confidence = 1.0

        return {
            "predicted_type": predicted,
            "confidence": round(confidence, 4),
            "all_probabilities": dict(sorted(proba.items(), key=lambda x: x[1], reverse=True)),
            "model_name": type(clf).__name__,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "predicted_type": "Unknown",
            "confidence": 0.0,
            "all_probabilities": {},
            "model_name": "Error",
            "error": str(e),
        }


def get_feature_importance(text: str, top_n: int = 10) -> List[Dict]:
    """Return top features responsible for prediction (Explainable AI)."""
    try:
        clf, tfidf, le = load_models()
        feature_names = tfidf.get_feature_names_out()

        X = tfidf.transform([text])
        tokens = X.toarray()[0]

        if hasattr(clf, "coef_"):
            pred_idx = clf.predict(X)[0]
            coefs = clf.coef_[pred_idx]
            # Multiply by token presence
            weighted = tokens * coefs
            top_indices = np.argsort(weighted)[::-1][:top_n]
            return [
                {"feature": feature_names[i], "weight": round(float(weighted[i]), 4)}
                for i in top_indices if weighted[i] > 0
            ]
        elif hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:top_n]
            return [
                {"feature": feature_names[i], "weight": round(float(importances[i]), 4)}
                for i in top_indices
            ]
        return []
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return []


def models_exist() -> bool:
    """Check if trained models exist on disk."""
    return (Path(CLASSIFIER_PATH).exists() and
            Path(TFIDF_PATH).exists() and
            Path(LABEL_ENCODER_PATH).exists())
