"""
models.py — ML Model persistence helpers.
Thin wrapper around ml_engine so app.py has a clean import.
"""

from ml_engine import (
    train_models,
    predict,
    models_exist,
    get_model_info,
)

__all__ = ["train_models", "predict", "models_exist", "get_model_info"]
