import random
import string
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


def generate_card_number():
    """Generates a unique 16-digit credit card number prefixed with 4532 (Visa-style)."""
    suffix = ''.join(random.choices(string.digits, k=12))
    return f'4532{suffix}'


def generate_account_number():
    """Generates a unique 10-digit account number."""
    return ''.join(random.choices(string.digits, k=10))


class UserProfile(models.Model):
    """
    Extended profile linked 1:1 to Django's built-in User model.
    Django's User already handles username, email, password, first/last name.
    This model stores everything else needed for the phone UI.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')

    # Personal info
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    date_of_birth = models.DateField()
    country = models.CharField(max_length=100)
    location = models.CharField(max_length=200, help_text='City / region used as home location baseline.')
    balance = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=100000.00,
        help_text='Account balance. Admin can update manually. Deducted automatically on transaction.'
    )

    # Banking info (auto-generated on signup)
    account_number = models.CharField(max_length=10, unique=True, blank=True)
    credit_card_number = models.CharField(max_length=16, unique=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.account_number:
            # Ensure uniqueness
            while True:
                num = generate_account_number()
                if not UserProfile.objects.filter(account_number=num).exists():
                    self.account_number = num
                    break
        if not self.credit_card_number:
            while True:
                num = generate_card_number()
                if not UserProfile.objects.filter(credit_card_number=num).exists():
                    self.credit_card_number = num
                    break
        super().save(*args, **kwargs)

    @property
    def age(self):
        from datetime import date
        today = date.today()
        dob = self.date_of_birth
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} — acct {self.account_number}'


class TransactionRecord(models.Model):
    """
    Every transaction submitted through the phone UI, including the
    fraud prediction result returned by the ML model. Doubles as
    transaction history for the velocity_last_24h auto-calculation.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        COMPLETED = 'completed', 'Completed'
        FLAGGED = 'flagged', 'Flagged as Fraud'
        BLOCKED = 'blocked', 'Blocked'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transaction_records')

    # Transaction inputs
    amount = models.FloatField()
    merchant_category = models.CharField(max_length=32)
    recipient_account = models.CharField(max_length=64, blank=True, null=True,
                                         help_text='Dummy field — not used in prediction.')
    transaction_country = models.CharField(max_length=100,
                                           help_text='Country selected by user at transaction time.')
    transaction_location = models.CharField(max_length=200,
                                            help_text='Location selected by user at transaction time.')
    is_personal_device = models.BooleanField(default=True,
                                             help_text='User toggle → maps to device_trust_score.')

    # Auto-derived at submission time (not filled by user)
    transaction_hour = models.IntegerField(help_text='Auto-detected from system time (0-23).')
    foreign_transaction = models.IntegerField(
        help_text='1 if transaction_country != signup country, else 0.')
    location_mismatch = models.IntegerField(
        help_text='1 if transaction_location != signup location, else 0.')
    device_trust_score = models.FloatField(
        help_text='85 if is_personal_device else 25.')
    velocity_last_24h = models.FloatField(
        help_text='Auto-counted from this user\'s TransactionRecord history.')
    cardholder_age = models.FloatField(
        help_text='Auto-derived from user profile date_of_birth.')

    # ML prediction result
    fraud_probability = models.FloatField(null=True, blank=True)
    is_fraud = models.IntegerField(null=True, blank=True)
    threshold_used = models.FloatField(null=True, blank=True)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['is_fraud']),
        ]

    def __str__(self):
        return f'{self.user.username} | ${self.amount} | fraud={self.is_fraud} | {self.created_at:%Y-%m-%d %H:%M}'


# ── Keep existing DatasetUpload / Transaction models below ──
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