"""
Feature engineering for the cleaned credit-card-fraud dataset.

Takes the output of clean_dataframe() (transactions/ml/cleaning.py) and:
  1. derives a handful of new signal-bearing features
  2. one-hot encodes merchant_category with a FIXED set of categories
     (important: a single transaction submitted later from the React
     form only has one category, so get_dummies on its own would
     silently drop columns — pd.Categorical with explicit `categories=`
     guarantees the same column set every time, whether you're
     engineering 10,000 rows or 1)
  3. returns (X, y) ready for the train/test split

Kept separate from cleaning.py because feature engineering depends on
already-clean data, but you might want to swap/extend the features
without touching the cleaning rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import TARGET_COLUMN

MERCHANT_CATEGORIES = ['Electronics', 'Travel', 'Grocery', 'Food', 'Clothing']


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Drop identifier/passthrough columns that aren't real features.
    # clean_dataframe() only touches the columns it knows about and
    # leaves everything else untouched, so if a source CSV/DB query
    # included transaction_id it would otherwise ride along into X here.
    df = df.drop(columns=['transaction_id'], errors='ignore')

    # --- derived features ---
    # amount is right-skewed (skew ~1.9) -> log-transform tames the tail
    df['log_amount'] = np.log1p(df['amount'])

    # fraud in this data clusters overnight (mean hour ~3.8 vs ~11.7 for
    # legit) — flag the late-night window directly rather than relying
    # on the model to learn it from a raw 0-23 integer.
    df['is_night'] = df['transaction_hour'].between(0, 5).astype(int)

    # cyclical encoding so hour 23 and hour 0 are recognized as adjacent
    df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)

    df['is_low_trust_device'] = (df['device_trust_score'] < 50).astype(int)
    df['is_high_velocity'] = (df['velocity_last_24h'] >= df['velocity_last_24h'].quantile(0.75)).astype(int)

    # the two strongest individual fraud signals in this data, combined
    df['foreign_and_mismatch'] = df['foreign_transaction'] * df['location_mismatch']

    # --- fixed-category one-hot encoding ---
    df['merchant_category'] = pd.Categorical(df['merchant_category'], categories=MERCHANT_CATEGORIES)
    df = pd.get_dummies(df, columns=['merchant_category'], prefix='merchant')
    dummy_cols = [c for c in df.columns if c.startswith('merchant_')]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # raw transaction_hour is now redundant with hour_sin/hour_cos/is_night
    df = df.drop(columns=['transaction_hour'])

    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else pd.Series(dtype=int, name=TARGET_COLUMN)
    X = df.drop(columns=[TARGET_COLUMN], errors='ignore')

    return X, y