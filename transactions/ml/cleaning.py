"""
Data cleaning rules for the credit-card-fraud dataset.

Plain pandas in, pandas out — no Django dependency — so this same
function can be reused by the management command below, a notebook, or
the training script without dragging the whole Django app along with it.

Rules (as specified for this project):
    * numerical / continuous columns -> fill NaN with the column MEDIAN
      (robust to outliers/skew; `amount` here is skewed ~1.9, so mean
      would get pulled toward the high-amount tail).
    * categorical columns with >2 levels -> fill NaN with the column
      MODE (most frequent category).
    * binary (0/1) columns -> fill NaN with 0.

This particular source CSV happens to ship with zero missing values,
but new data uploaded later (or submitted through the React form) won't
necessarily be that clean, so these rules stay in place rather than
being skipped.
"""
from __future__ import annotations

import pandas as pd

NUMERICAL_COLUMNS = [
    'amount',
    'transaction_hour',
    'device_trust_score',
    'velocity_last_24h',
    'cardholder_age',
]

CATEGORICAL_COLUMNS = [
    'merchant_category',  
]

BINARY_COLUMNS = [
    'foreign_transaction',
    'location_mismatch',
]

TARGET_COLUMN = 'is_fraud'

ALL_FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS


def compute_impute_values(df: pd.DataFrame) -> dict:
    """
    Computes the median/mode/0 fill values from a (presumably large,
    representative) DataFrame — typically the full training set — so
    they can be saved and reused later.

    This matters for single-row inference: clean_dataframe() computes
    medians/modes from whatever DataFrame it's given, which is correct
    for batch cleaning but breaks for a single submitted transaction —
    the "median" of one row's possibly-missing value is just that
    missing value, nothing to fill from. Saving training-time stats
    once and reusing them at inference time fixes that.
    """
    values = {}
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            values[col] = float(df[col].median())
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            mode_value = df[col].mode(dropna=True)
            values[col] = mode_value.iloc[0] if not mode_value.empty else 'Unknown'
    for col in BINARY_COLUMNS:
        values[col] = 0
    return values


def apply_impute_values(df: pd.DataFrame, impute_values: dict) -> pd.DataFrame:
    """Fills missing values using precomputed stats from compute_impute_values(), instead of recomputing from df itself."""
    df = df.copy()
    for col, fill_value in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a *new* DataFrame with missing values imputed according to
    the column rules above. Does not mutate the input in place.
    """
    df = df.copy()

    missing_before = df[ALL_FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()

    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            mode_value = df[col].mode(dropna=True)
            mode_value = mode_value.iloc[0] if not mode_value.empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)

    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # is_fraud should never be missing in practice, but guard anyway —
    # an unlabeled row is useless for supervised training, so we drop it
    # rather than guess a label.
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    missing_after = df[ALL_FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
    cleaning_report = pd.DataFrame({
        'missing_before': missing_before,
        'missing_after': missing_after,
    })
    df.attrs['cleaning_report'] = cleaning_report

    return df