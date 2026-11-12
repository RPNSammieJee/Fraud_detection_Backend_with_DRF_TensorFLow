"""
Train/validation/test split, scaling, and SMOTE for the fraud-detection
dataset.

Order matters a lot here, and it's the most common place this kind of
pipeline leaks information (or lies to you about performance):

  1. SPLIT TEST OFF FIRST (stratified on is_fraud) — nothing downstream
     is allowed to see it until final evaluation.
  2. SPLIT TRAIN/VALIDATION (stratified) — both still real, original
     data at this point.
  3. SCALE — fit StandardScaler on the TRAIN split only, then reuse
     that *same fitted scaler* to transform validation and test.
     Fitting on the full dataset before splitting would leak
     validation/test statistics (mean/std) into training.
  4. SMOTE — applied to the *scaled training data only*. Validation and
     test both stay 100% real, untouched, original data at their real
     ~10.3% fraud rate.

Why validation has to be real (not SMOTE'd): validation is what you
watch during training (early stopping, "is this still improving"). If
it contains synthetic SMOTE samples, you're measuring "does the model
fit synthetic data well" instead of "does the model generalize to real
transactions" — and you'd stop training at the wrong point.
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Only scale genuinely continuous columns. Binary flags and one-hot dummy
# columns are already on a 0/1 scale and StandardScaler would just add
# noise (and confusion) for no benefit.
CONTINUOUS_COLUMNS_TO_SCALE = [
    'amount', 'log_amount', 'device_trust_score', 'velocity_last_24h',
    'cardholder_age', 'hour_sin', 'hour_cos',
]


def split_scale_smote(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    test_size: fraction of the FULL dataset held out as the final test set.
    val_size: fraction of the remaining (post-test) data held out as validation.

    Returns: X_train_resampled, X_val_scaled, X_test_scaled,
             y_train_resampled, y_val, y_test,
             report (dict of class counts at each stage),
             artifacts (dict with fitted scaler + exact feature column order).
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size,
        random_state=random_state, stratify=y_train_full,
    )

    scale_cols = [c for c in CONTINUOUS_COLUMNS_TO_SCALE if c in X.columns]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    report = {
        'train_before_smote': y_train.value_counts().sort_index().to_dict(),
        'train_after_smote': y_train_resampled.value_counts().sort_index().to_dict(),
        'val_unchanged': y_val.value_counts().sort_index().to_dict(),
        'test_unchanged': y_test.value_counts().sort_index().to_dict(),
    }

    artifacts = {
        'scaler': scaler,
        'scaled_columns': scale_cols,
        'feature_columns': list(X.columns),
    }

    return (
        X_train_resampled, X_val_scaled, X_test_scaled,
        y_train_resampled, y_val, y_test,
        report, artifacts,
    )
