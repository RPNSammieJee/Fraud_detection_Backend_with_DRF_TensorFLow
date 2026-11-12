from rest_framework import serializers

from .models import DatasetUpload, Transaction

MERCHANT_CATEGORY_CHOICES = ['Electronics', 'Travel', 'Grocery', 'Food', 'Clothing']


class DatasetUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetUpload
        fields = ['id', 'file', 'uploaded_at', 'status', 'row_count', 'error_message']
        read_only_fields = ['uploaded_at', 'status', 'row_count', 'error_message']


class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = '__all__'


class PredictionInputSerializer(serializers.Serializer):
    """
    Validates a single transaction coming from the React "test the model"
    form. Used by the /api/predict/ endpoint (Phase 3/4). Kept here so the
    frontend/backend contract is settled before the form gets built.
    """
    amount = serializers.FloatField(min_value=0)
    transaction_hour = serializers.IntegerField(min_value=0, max_value=23)
    merchant_category = serializers.ChoiceField(choices=MERCHANT_CATEGORY_CHOICES)
    foreign_transaction = serializers.ChoiceField(choices=[0, 1])
    location_mismatch = serializers.ChoiceField(choices=[0, 1])
    device_trust_score = serializers.FloatField(min_value=0, max_value=100)
    velocity_last_24h = serializers.FloatField(min_value=0)
    cardholder_age = serializers.FloatField(min_value=0, max_value=120)
