"""
Evaluation utilities for the fraud detector.

Deliberately framework-agnostic: every function here takes plain numpy
arrays (y_true, y_pred_proba), so it works whether those probabilities
came from the Keras model, a sanity-check sklearn model, or anything
else. Keeps this module testable without needing TensorFlow installed.

Why these specific metrics, and why threshold tuning matters:
With ~10.3% fraud in real data, a model that predicts "never fraud" is
89.7% accurate and catches zero fraud. Accuracy is close to useless
here. What matters:
  - Recall: of the actual fraud cases, how many did we catch?
  - Precision: of the cases we flagged as fraud, how many really were?
  - ROC-AUC / PR-AUC: threshold-independent measures of separability
    (PR-AUC is the more informative one on imbalanced data).
  - The default 0.5 threshold is arbitrary. Fraud teams usually care
    more about recall (missing fraud is expensive) but not so much that
    every legitimate customer gets flagged — so we scan thresholds and
    report the one that maximizes F1 as a reasonable starting point,
    alongside 0.5, so you can see the tradeoff and pick what fits your
    actual cost of a false positive vs a false negative.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')  # no display server in this environment
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def metrics_at_threshold(y_true, y_pred_proba, threshold: float) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'threshold': float(threshold),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def find_best_threshold(y_true, y_pred_proba, metric: str = 'f1') -> dict:
    """Scans thresholds from the precision-recall curve, returns metrics at the one maximizing F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # precision_recall_curve returns one more point than thresholds (the
    # last precision/recall pair has no corresponding threshold) — drop it.
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    return metrics_at_threshold(y_true, y_pred_proba, best_threshold)


def evaluate_model(y_true, y_pred_proba) -> dict:
    """Full evaluation report: threshold-independent metrics + metrics at 0.5 and at the best-F1 threshold."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba).ravel()

    report = {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'at_threshold_0.5': metrics_at_threshold(y_true, y_pred_proba, 0.5),
        'at_best_f1_threshold': find_best_threshold(y_true, y_pred_proba),
        'positive_rate_in_data': float(y_true.mean()),
    }
    return report


def save_metrics_json(report: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


# --- plots -----------------------------------------------------------------

def plot_training_history(history: dict, output_path: str) -> None:
    """history: a Keras History.history dict (keys like 'loss', 'val_loss', 'auc', 'val_auc', ...)."""
    metrics_to_plot = [m for m in ('loss', 'auc', 'pr_auc') if m in history]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        ax.plot(history[metric], label=f'train {metric}')
        val_key = f'val_{metric}'
        if val_key in history:
            ax.plot(history[val_key], label=f'val {metric}')
        ax.set_xlabel('epoch')
        ax.set_title(metric)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm_dict: dict, output_path: str) -> None:
    cm = np.array([[cm_dict['tn'], cm_dict['fp']], [cm_dict['fn'], cm_dict['tp']]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Legit', 'Predicted: Fraud'])
    ax.set_yticklabels(['Actual: Legit', 'Actual: Fraud'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_roc_and_pr_curves(y_true, y_pred_proba, output_path: str) -> None:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = y_true.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_true, y_pred_proba):.3f}')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()

    ax2.plot(recalls, precisions, label=f'PR-AUC = {average_precision_score(y_true, y_pred_proba):.3f}')
    ax2.axhline(baseline, linestyle='--', color='gray', label=f'baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
