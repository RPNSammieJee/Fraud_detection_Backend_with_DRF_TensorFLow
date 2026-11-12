"""
Single-transaction inference for the fraud detector.

This is the piece that makes "train once, predict forever" actually
safe: a transaction submitted from the React form has to go through
*exactly* the same cleaning -> feature engineering -> scaling steps the
training data went through, in the same order, producing the same
column set in the same order — otherwise the model is silently fed
garbage that happens to be the right shape.

Loads three artifacts, all produced earlier in the pipeline:
    media/model/fraud_model.keras        <- from `train_model`
    media/processed/scaler.joblib        <- from `prepare_training_data`
    media/processed/feature_columns.json <- from `prepare_training_data`

Cached as module-level globals so the model/scaler are loaded from disk
once per process, not on every request.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from django.conf import settings

from .cleaning import apply_impute_values
from .feature_engineering import engineer_features

_model = None
_scaler = None
_feature_columns = None
_scaled_columns = None
_impute_values = None


class ModelNotReadyError(Exception):
    """Raised when the trained model / scaler / feature spec aren't on disk yet."""


def _load_artifacts():
    global _model, _scaler, _feature_columns, _scaled_columns, _impute_values

    if _model is not None:
        return  # already loaded

    model_path = settings.MEDIA_ROOT / 'model' / 'fraud_model.keras'
    scaler_path = settings.MEDIA_ROOT / 'processed' / 'scaler.joblib'
    columns_path = settings.MEDIA_ROOT / 'processed' / 'feature_columns.json'
    impute_path = settings.MEDIA_ROOT / 'processed' / 'impute_values.json'

    missing = [str(p) for p in (model_path, scaler_path, columns_path, impute_path) if not p.exists()]
    if missing:
        raise ModelNotReadyError(
            f'Missing trained-model artifacts: {missing}. '
            'Run `python manage.py prepare_training_data` then `python manage.py train_model` first.'
        )

    from tensorflow import keras  # imported lazily so the rest of the app works without TF installed

    _model = keras.models.load_model(model_path)
    _scaler = joblib.load(scaler_path)
    with open(columns_path) as f:
        spec = json.load(f)
    _feature_columns = spec['feature_columns']
    _scaled_columns = spec['scaled_columns']
    with open(impute_path) as f:
        _impute_values = json.load(f)


def predict_single(transaction: dict) -> dict:
    """
    transaction: dict matching PredictionInputSerializer's validated_data,
    e.g. {'amount': 84.47, 'transaction_hour': 22, 'merchant_category':
    'Electronics', 'foreign_transaction': 0, 'location_mismatch': 0,
    'device_trust_score': 66, 'velocity_last_24h': 3, 'cardholder_age': 40}

    Returns: {'fraud_probability': float, 'is_fraud': int, 'threshold_used': float}
    """
    _load_artifacts()

    df = pd.DataFrame([transaction])
    # Fill any missing fields using the TRAINING-TIME medians/modes
    # (saved in impute_values.json), not stats recomputed from this
    # single row — a "median" of one possibly-missing value is
    # meaningless, which is exactly the bug this avoids.
    cleaned = apply_impute_values(df, _impute_values)

    X, _ = engineer_features(cleaned)

    # Reindex to the exact training-time column set/order. Any column
    # the model expects but this single row didn't produce (e.g. a
    # merchant category that happens to match every fixed dummy) gets
    # filled with 0 rather than raising — fixed-category encoding in
    # feature_engineering.py should already prevent this, but this is
    # the safety net if the model is ever retrained with a different
    # feature set without redeploying this code.
    X = X.reindex(columns=_feature_columns, fill_value=0)

    X_scaled = X.copy()
    X_scaled[_scaled_columns] = _scaler.transform(X[_scaled_columns])

    probability = float(_model.predict(X_scaled.values, verbose=0).ravel()[0])

    # 0.5 is the conservative/default cutoff. Phase 5's evaluation found
    # a higher threshold (~0.98) gives much better precision at the cost
    # of recall — swap this constant for that value once you've decided
    # which tradeoff fits your product.
    threshold = 0.5
    return {
        'fraud_probability': probability,
        'is_fraud': int(probability >= threshold),
        'threshold_used': threshold,
    }