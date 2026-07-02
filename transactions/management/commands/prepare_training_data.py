"""
python manage.py prepare_training_data
python manage.py prepare_training_data --input media/cleaned/cleaned_transactions.csv --test-size 0.2 --val-size 0.15

Runs feature engineering, stratified train/validation/test split,
scaling, and SMOTE. Writes everything Phase 3 (neural network training)
needs into media/processed/:

    X_train.csv, y_train.csv   <- SMOTE-balanced, scaled  (train on this)
    X_val.csv,   y_val.csv     <- real data, scaled, never touched by SMOTE
                                   (watch this during training / early stopping)
    X_test.csv,  y_test.csv    <- real data, scaled, never touched by SMOTE,
                                   never looked at until final evaluation
    scaler.joblib               <- fit on train only; the /predict/ endpoint
                                    reuses this exact scaler at inference time
    feature_columns.json        <- exact column order + which columns were
                                    scaled, so a single new transaction can be
                                    engineered + reindexed to match
"""
import json

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from transactions.ml.cleaning import compute_impute_values
from transactions.ml.feature_engineering import engineer_features
from transactions.ml.prepare_data import split_scale_smote


class Command(BaseCommand):
    help = 'Feature-engineer, split (train/val/test), scale, and SMOTE-balance the cleaned dataset.'

    def add_arguments(self, parser):
        parser.add_argument('--input', type=str, default=None)
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--val-size', type=float, default=0.15)

    def handle(self, *args, **options):
        input_path = options['input'] or (settings.MEDIA_ROOT / 'cleaned' / 'cleaned_transactions.csv')
        df = pd.read_csv(input_path)
        self.stdout.write(f'Loaded {len(df)} cleaned rows from {input_path}')

        # Saved so single-row predictions at inference time can reuse
        # these exact stats instead of recomputing a "median" from a
        # batch of one (see transactions/ml/cleaning.py for why that breaks).
        impute_values = compute_impute_values(df)

        X, y = engineer_features(df)
        self.stdout.write(f'Engineered {X.shape[1]} features:')
        self.stdout.write(f'  {list(X.columns)}')

        X_train, X_val, X_test, y_train, y_val, y_test, report, artifacts = split_scale_smote(
            X, y, test_size=options['test_size'], val_size=options['val_size'],
        )

        self.stdout.write(self.style.WARNING('\nClass balance at each stage:'))
        self.stdout.write(f"  Train, before SMOTE : {report['train_before_smote']}")
        self.stdout.write(f"  Train, after SMOTE  : {report['train_after_smote']}")
        self.stdout.write(f"  Validation (real)   : {report['val_unchanged']}")
        self.stdout.write(f"  Test (real)         : {report['test_unchanged']}")

        out_dir = settings.MEDIA_ROOT / 'processed'
        out_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(out_dir / 'X_train.csv', index=False)
        y_train.to_csv(out_dir / 'y_train.csv', index=False)
        X_val.to_csv(out_dir / 'X_val.csv', index=False)
        y_val.to_csv(out_dir / 'y_val.csv', index=False)
        X_test.to_csv(out_dir / 'X_test.csv', index=False)
        y_test.to_csv(out_dir / 'y_test.csv', index=False)

        joblib.dump(artifacts['scaler'], out_dir / 'scaler.joblib')
        with open(out_dir / 'feature_columns.json', 'w') as f:
            json.dump({
                'feature_columns': artifacts['feature_columns'],
                'scaled_columns': artifacts['scaled_columns'],
            }, f, indent=2)
        with open(out_dir / 'impute_values.json', 'w') as f:
            json.dump(impute_values, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f'\nWrote train/val/test sets + scaler + feature_columns.json to {out_dir}'))
        self.stdout.write(f'  X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {X_test.shape}')