from django.db import models


class DatasetUpload(models.Model):
    """
    One row per CSV file someone uploads through the admin or the
    /api/datasets/upload/ endpoint. Keeps an audit trail of what was
    uploaded, when, and whether it parsed cleanly.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    row_count = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f'Dataset #{self.pk} ({self.status}) - {self.row_count} rows'


class Transaction(models.Model):
    """
    One transaction record. Field set mirrors the credit-card-fraud CSV
    (transaction_id, amount, transaction_hour, merchant_category,
    foreign_transaction, location_mismatch, device_trust_score,
    velocity_last_24h, cardholder_age, is_fraud).

    Nullable on every feature column on purpose: raw uploaded data may
    have missing values, and the cleaning step (see
    transactions/ml/cleaning.py) is what fills them in — the DB should
    reflect what was actually uploaded, not a pre-cleaned version of it.

    Column roles for cleaning/feature engineering downstream:
      - numerical (continuous)  -> median imputation
          amount, transaction_hour, device_trust_score,
          velocity_last_24h, cardholder_age
      - categorical (>2 levels) -> mode imputation
          merchant_category (Electronics/Travel/Grocery/Food/Clothing)
      - binary (0/1)            -> fillna(0)
          foreign_transaction, location_mismatch
      - target
          is_fraud
    """

    dataset = models.ForeignKey(
        DatasetUpload,
        on_delete=models.CASCADE,
        related_name='transactions',
        null=True,
        blank=True,
        help_text='Which uploaded CSV this row came from.',
    )

    source_transaction_id = models.CharField(
        max_length=64, null=True, blank=True,
        help_text='transaction_id from the source CSV, kept for traceability only — not a model feature.',
    )

    # --- numerical / continuous ---
    amount = models.FloatField(null=True, blank=True)
    transaction_hour = models.FloatField(null=True, blank=True)  # 0-23
    device_trust_score = models.FloatField(null=True, blank=True)  # 0-99
    velocity_last_24h = models.FloatField(null=True, blank=True)
    cardholder_age = models.FloatField(null=True, blank=True)

    # --- categorical (more than 2 levels) ---
    merchant_category = models.CharField(max_length=32, null=True, blank=True)

    # --- binary flags ---
    foreign_transaction = models.FloatField(null=True, blank=True)
    location_mismatch = models.FloatField(null=True, blank=True)

    # --- target label ---
    is_fraud = models.IntegerField(choices=[(0, 'Legitimate'), (1, 'Fraud')])

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_fraud']),
            models.Index(fields=['dataset']),
        ]

    def __str__(self):
        return f'Txn #{self.pk} | amount={self.amount} | fraud={self.is_fraud}'
