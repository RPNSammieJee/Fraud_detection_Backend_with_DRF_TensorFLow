from django.urls import path

from .views import DatasetListView, DatasetUploadView, TransactionListView, PredictView

urlpatterns = [
    path('datasets/upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('datasets/', DatasetListView.as_view(), name='dataset-list'),
    path('transactions/', TransactionListView.as_view(), name='transaction-list'),
    path('predict/', PredictView.as_view(), name='predict'), 
]
