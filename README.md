# Fraud Detection — Backend

Django REST Framework backend for the fraud detection project: upload
the dataset, store it in the DB, browse it in the admin, clean it,
engineer features, and prepare it for model training.

## Dataset

Credit-card-fraud CSV, 10,000 rows, **no missing values**, fraud rate
**1.51%** (151 fraud / 9849 legit). Verified with a RandomForest sanity
check before building around it: ROC-AUC ≈ 1.0 — strong, genuine signal,
driven mainly by `foreign_transaction`, `location_mismatch`,
`device_trust_score`, and `velocity_last_24h` (fraud transactions
average a device trust score of ~38 vs ~62 for legit ones, and cluster
heavily overnight — mean hour ~3.8 vs ~11.7).

| Column | Type | Cleaning rule |
|---|---|---|
| amount | numerical | median (skew ~1.9, hence median not mean) |
| transaction_hour | numerical (0–23) | median |
| device_trust_score | numerical (0–99) | median |
| velocity_last_24h | numerical | median |
| cardholder_age | numerical | median |
| merchant_category | categorical (5 levels: Electronics/Travel/Grocery/Food/Clothing) | mode |
| foreign_transaction | binary | fill 0 |
| location_mismatch | binary | fill 0 |
| is_fraud | target | — |

This source file ships clean (zero NaNs), but the cleaning rules stay
in the pipeline regardless — new uploads or single transactions
submitted later from the React form won't necessarily be that tidy.

## Setup

```bash
cd fraud_detection_backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser   # this is your admin login
python manage.py runserver
```

Admin panel: http://127.0.0.1:8000/admin/

## Loading your data

```bash
curl.exe -X POST http://127.0.0.1:8000/api/datasets/upload/ `
  -F "file=@C:\Users\hp\Downloads\credit_card_fraud_10k.csv"
```

Parses the CSV with pandas, creates a `DatasetUpload` row, and
bulk-creates one `Transaction` row per record. `transaction_id` from
the source CSV is kept as `source_transaction_id` for traceability only
— it is explicitly excluded from the feature set later (an incrementing
ID would otherwise leak as a spurious "feature").

## Cleaning the data

```bash
python manage.py clean_dataset
# or, to clean just one upload:
python manage.py clean_dataset --dataset 1
```

Prints a before/after missing-value report and writes
`media/cleaned/cleaned_transactions.csv`.

## Feature engineering, split, scaling, SMOTE

```bash
python manage.py prepare_training_data
```

1. **Feature engineering** (`transactions/ml/feature_engineering.py`)
   — derives `log_amount` (tames the right skew), `is_night` (flags the
   0–5am window, where fraud is concentrated), `hour_sin`/`hour_cos`
   (cyclical encoding so hour 23 and hour 0 are recognized as
   adjacent), `is_low_trust_device`, `is_high_velocity`, and
   `foreign_and_mismatch` (interaction of the two strongest individual
   fraud signals). `merchant_category` is one-hot encoded with a
   **fixed category list** — important because a single transaction
   from the React form only has one category, so a naive
   `pd.get_dummies` would silently produce fewer columns than the model
   expects. Final feature count: **18 columns**.
2. **3-way stratified split** (test 20% / val 15% of remainder / train
   the rest) — all three keep the same ~1.51% fraud rate before
   anything else happens to them.
3. **Scaling** — `StandardScaler` fit on train only, reused (not refit)
   on validation and test.
4. **SMOTE** — applied to scaled training data only. Validation and
   test both stay 100% real, untouched, original imbalance — validation
   is what you watch during training (early stopping), and if it
   contained synthetic samples you'd be measuring "does the model fit
   fake data" instead of "does it generalize."

Verified end-to-end against the real file:

| Stage | Class 0 (legit) | Class 1 (fraud) |
|---|---|---|
| Train, before SMOTE | 6697 | 103 |
| Train, after SMOTE | 6697 | 6697 |
| Validation (real, untouched) | 1182 | 18 (1.5%) |
| Test (real, untouched) | 1970 | 30 (1.5%) |

Output, all in `media/processed/`:

| File | Contents |
|---|---|
| `X_train.csv`, `y_train.csv` | SMOTE-balanced, scaled — train on this |
| `X_val.csv`, `y_val.csv` | real, scaled, untouched — watch during training |
| `X_test.csv`, `y_test.csv` | real, scaled, untouched — final evaluation only |
| `scaler.joblib` | fitted scaler, reused at `/api/predict/` inference time |
| `feature_columns.json` | exact column order + which columns were scaled |

## Neural network (`transactions/ml/model.py`)

Small MLP (2 hidden layers: 32→16, ReLU, batch norm, dropout 0.3),
binary cross-entropy, Adam. Compiled with ROC-AUC, PR-AUC, precision,
and recall as tracked metrics — not accuracy, since with a 1.5% fraud
rate a model that never predicts fraud would already be 98.5%
"accurate" while catching nothing. Kept deliberately small: 18 input
features and ~13k rows after SMOTE don't need a deep/wide network — a
bigger one would just memorize.

Trained on SMOTE-balanced data (no `class_weight` on top — combining
both tends to over-correct and tank precision), validated/evaluated on
real, imbalanced data.

## Training the model

```bash
python manage.py train_model
# or tune it:
python manage.py train_model --epochs 150 --batch-size 32 --patience 15
```

This is the only command in the project that needs TensorFlow — every
earlier step is plain pandas/sklearn/imblearn, so don't worry about
installing TF unless you're getting to this step.

What it does:
- Loads `X_train`/`y_train` (SMOTE-balanced) and `X_val`/`y_val` (real,
  untouched) from `media/processed/`.
- Trains with **early stopping on validation PR-AUC** (not val_loss or
  val_accuracy — PR-AUC is the metric that actually reflects
  performance on imbalanced data) and a learning-rate-reduction
  callback on plateau.
- Restores the best-validation-PR-AUC epoch's weights automatically.
- Evaluates once, at the end, on the **held-out test set** — data the
  model has never seen during training or validation.

Saves to `media/model/`:

| File | Contents |
|---|---|
| `fraud_model.keras` | the trained model |
| `training_history.json` / `.png` | per-epoch loss/AUC/PR-AUC/precision/recall, train vs val |
| `test_metrics.json` | full evaluation report: ROC-AUC, PR-AUC, confusion matrix, precision/recall/F1 at 0.5 and at the best-F1 threshold |
| `confusion_matrix.png`, `roc_pr_curves.png` | plots |

### Validated pipeline results

I couldn't install TensorFlow in the sandbox used to build this (no
network access), so I validated the full training → evaluation →
plotting pipeline against your real data using a comparable small
neural net (scikit-learn's MLPClassifier, same 32→16 ReLU shape) as a
stand-in for the exact Keras model. This confirms the *pipeline* is
correct; your actual numbers from `fraud_model.keras` may differ
slightly from these but should land in a similar range:

```
Validation: ROC-AUC 0.998   PR-AUC 0.915
Test:       ROC-AUC 0.995   PR-AUC 0.854

At threshold 0.5:       precision 0.63   recall 0.87   (26/30 fraud caught, 15 false alarms)
At best-F1 threshold:   precision 0.88   recall 0.73   (22/30 fraud caught,  3 false alarms)
```

That threshold comparison is the real takeaway: the default 0.5 cutoff
catches more fraud but triples your false-alarm rate. Worth deciding
deliberately which tradeoff fits — flag more and annoy more legitimate
customers, or flag less and miss some fraud. `test_metrics.json` has
both so you can pick.

## Evaluation (`transactions/ml/evaluation.py`)

Framework-agnostic (works on any `y_true`/`y_pred_proba` arrays, not
just Keras output). Computes ROC-AUC, PR-AUC, confusion matrix,
precision/recall/F1 at both the default 0.5 threshold and the
threshold that maximizes F1. Also generates training-history,
confusion-matrix, and ROC/PR-curve plots — all three confirmed
rendering correctly against real-shaped data during testing.

## Endpoints so far

| Method | URL | Purpose |
|---|---|---|
| POST | `/api/datasets/upload/` | Upload a CSV, parse + store rows |
| GET | `/api/datasets/` | List upload history |
| GET | `/api/transactions/` | Paginated list of rows, `?dataset=<id>` to filter |
| * | `/admin/` | Django admin — browse/search/filter everything |

## Roadmap

1. ~~Feature engineering~~ ✅
2. ~~Train/validation/test split~~ ✅
3. ~~Scaling~~ ✅
4. ~~SMOTE~~ ✅
5. ~~Train the neural network + evaluate~~ ✅
6. **`/api/predict/` endpoint** — loads the saved model + scaler,
   accepts a single transaction (schema already drafted in
   `PredictionInputSerializer`), runs it through the same feature
   engineering + scaling, returns fraud probability + label.
7. **React frontend** — form mirroring `PredictionInputSerializer`,
   calls `/api/predict/`, shows the result with the kind of UI you
   asked for from the start (clean interaction, real-time feedback).

Say the word and I'll move on to step 6 — the prediction endpoint that
wires the trained model up to the API.
