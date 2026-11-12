from django.contrib import admin
from .models import DatasetUpload, Transaction, UserProfile, TransactionRecord


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ('user', 'account_number', 'credit_card_number', 'country', 'location', 'age', 'created_at')
    search_fields = ('user__username', 'user__email', 'account_number', 'credit_card_number')
    readonly_fields = ('account_number', 'credit_card_number', 'created_at', 'updated_at')


@admin.register(TransactionRecord)
class TransactionRecordAdmin(admin.ModelAdmin):
    list_display  = ('id', 'user', 'amount', 'merchant_category', 'is_fraud',
                     'fraud_probability', 'status', 'created_at')
    list_filter   = ('is_fraud', 'status', 'merchant_category', 'foreign_transaction', 'location_mismatch')
    search_fields = ('user__username',)
    readonly_fields = ('fraud_probability', 'is_fraud', 'threshold_used',
                       'transaction_hour', 'foreign_transaction', 'location_mismatch',
                       'device_trust_score', 'velocity_last_24h', 'cardholder_age', 'created_at')
    ordering = ('-created_at',)


@admin.register(DatasetUpload)
class DatasetUploadAdmin(admin.ModelAdmin):
    list_display    = ('id', 'file', 'status', 'row_count', 'uploaded_at')
    list_filter     = ('status',)
    readonly_fields = ('status', 'row_count', 'error_message', 'uploaded_at')
    ordering        = ('-uploaded_at',)


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display  = ('id', 'dataset', 'amount', 'merchant_category',
                     'transaction_hour', 'is_fraud', 'created_at')
    list_filter   = ('is_fraud', 'merchant_category', 'foreign_transaction', 'location_mismatch')
    search_fields = ('id', 'source_transaction_id')
    list_per_page = 50
    ordering      = ('-created_at',)
