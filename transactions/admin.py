from django.contrib import admin

from .models import DatasetUpload, Transaction


@admin.register(DatasetUpload)
class DatasetUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'status', 'row_count', 'uploaded_at')
    list_filter = ('status',)
    readonly_fields = ('status', 'row_count', 'error_message', 'uploaded_at')
    ordering = ('-uploaded_at',)


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'dataset', 'amount', 'merchant_category', 'transaction_hour',
        'foreign_transaction', 'location_mismatch', 'device_trust_score',
        'is_fraud', 'created_at',
    )
    list_filter = ('is_fraud', 'merchant_category', 'foreign_transaction', 'location_mismatch')
    search_fields = ('id', 'source_transaction_id')
    list_per_page = 50
    ordering = ('-created_at',)
