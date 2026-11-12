"""
python manage.py train_model
python manage.py train_model --epochs 100 --batch-size 64

Loads the train/val/test sets produced by `prepare_training_data`,
builds the MLP from transactions/ml/model.py, trains with early
stopping (watching validation PR-AUC — the right metric to watch on
imbalanced data, not val_loss or val_accuracy), evaluates on the
held-out test set, and saves everything to media/model/:

    fraud_model.keras       <- the trained model (best epoch, restored
                                automatically by EarlyStopping's
                                restore_best_weights)
    training_history.json   <- per-epoch loss/auc/pr_auc/precision/recall
                                for train and validation
    training_history.png    <- plotted curves
    test_metrics.json       <- full evaluation report on the test set
                                (ROC-AUC, PR-AUC, confusion matrix,
                                precision/recall/F1 at 0.5 and at the
                                best-F1 threshold)
    confusion_matrix.png
    roc_pr_curves.png

This is the only command in the project that needs TensorFlow
installed (everything before it is plain pandas/sklearn/imblearn).
"""
import json

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from transactions.ml.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_training_history,
    save_metrics_json,
)
from transactions.ml.model import build_model


class Command(BaseCommand):
    help = 'Train the fraud-detection neural network on the prepared data and evaluate it on the test set.'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--patience', type=int, default=10, help='Early stopping patience, in epochs.')

    def handle(self, *args, **options):
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise CommandError(
                'TensorFlow is not installed. Run `pip install -r requirements.txt` '
                '(this is the one step in the project that needs it).'
            ) from exc

        processed_dir = settings.MEDIA_ROOT / 'processed'
        required = ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv']
        missing = [f for f in required if not (processed_dir / f).exists()]
        if missing:
            raise CommandError(
                f'Missing prepared-data files {missing} in {processed_dir}. '
                'Run `python manage.py prepare_training_data` first.'
            )

        X_train = pd.read_csv(processed_dir / 'X_train.csv')
        y_train = pd.read_csv(processed_dir / 'y_train.csv').iloc[:, 0]
        X_val = pd.read_csv(processed_dir / 'X_val.csv')
        y_val = pd.read_csv(processed_dir / 'y_val.csv').iloc[:, 0]
        X_test = pd.read_csv(processed_dir / 'X_test.csv')
        y_test = pd.read_csv(processed_dir / 'y_test.csv').iloc[:, 0]

        self.stdout.write(
            f'Loaded train={X_train.shape}, val={X_val.shape}, test={X_test.shape}'
        )

        model = build_model(input_dim=X_train.shape[1])
        model.summary(print_fn=self.stdout.write)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_pr_auc', mode='max',
                patience=options['patience'], restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_pr_auc', mode='max', factor=0.5, patience=5, min_lr=1e-6,
            ),
        ]

        history = model.fit(
            X_train.values, y_train.values,
            validation_data=(X_val.values, y_val.values),
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            callbacks=callbacks,
            verbose=2,
        )

        out_dir = settings.MEDIA_ROOT / 'model'
        out_dir.mkdir(parents=True, exist_ok=True)

        model.save(out_dir / 'fraud_model.keras')

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        plot_training_history(history_dict, str(out_dir / 'training_history.png'))

        self.stdout.write(self.style.WARNING('\nEvaluating on the held-out test set (never seen during training)...'))
        test_proba = model.predict(X_test.values, verbose=0).ravel()
        test_report = evaluate_model(y_test, test_proba)

        save_metrics_json(test_report, str(out_dir / 'test_metrics.json'))
        plot_confusion_matrix(test_report['at_threshold_0.5']['confusion_matrix'], str(out_dir / 'confusion_matrix.png'))
        plot_roc_and_pr_curves(y_test, test_proba, str(out_dir / 'roc_pr_curves.png'))

        self.stdout.write(self.style.SUCCESS(f'\n=== TEST SET RESULTS ==='))
        self.stdout.write(f"ROC-AUC: {test_report['roc_auc']:.4f}")
        self.stdout.write(f"PR-AUC:  {test_report['pr_auc']:.4f}")
        self.stdout.write(f"\nAt threshold 0.5:")
        self.stdout.write(f"  {test_report['at_threshold_0.5']}")
        self.stdout.write(f"\nAt best-F1 threshold:")
        self.stdout.write(f"  {test_report['at_best_f1_threshold']}")
        self.stdout.write(self.style.SUCCESS(f'\nSaved model + metrics + plots to {out_dir}'))
