"""
Neural network architecture for fraud detection.

A small MLP is the right call here, not a deep/wide net: 18 input
features, ~13k rows after SMOTE — a big network would just memorize.
Regularization (dropout + batch norm) matters more than depth, and with
this little data a (32, 16) network is already plenty of capacity;
(64, 32, 16) would be overkill and more prone to overfitting.

Trained on SMOTE-balanced data (50/50), so the loss function doesn't
need class_weight on top of that — doing both at once tends to
over-correct and tank precision. Validated/evaluated on real,
imbalanced data (~1.5% fraud), which is where the metrics that matter
(precision/recall/AUC) actually come from.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (32, 16),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name='features')
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    outputs = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)

    model = keras.Model(inputs, outputs, name='fraud_detector')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='pr_auc', curve='PR'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )
    return model
