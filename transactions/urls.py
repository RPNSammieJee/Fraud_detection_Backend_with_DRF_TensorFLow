from django.urls import path

from .views import DatasetListView, DatasetUploadView, PredictView, TransactionListView
from .views_auth import (
    LoginView,
    MakeTransactionView,
    ProfileView,
    SignupView,
    TransactionHistoryView,
)

urlpatterns = [
    # ── Existing dataset / predict endpoints (unchanged) ─────────────
    path('datasets/upload/',  DatasetUploadView.as_view(),   name='dataset-upload'),
    path('datasets/',         DatasetListView.as_view(),     name='dataset-list'),
    path('transactions/',     TransactionListView.as_view(), name='transaction-list'),
    path('predict/',          PredictView.as_view(),         name='predict'),

    # ── Auth ─────────────────────────────────────────────────────────
    path('auth/signup/',      SignupView.as_view(),          name='signup'),
    path('auth/login/',       LoginView.as_view(),           name='login'),
    path('auth/profile/',     ProfileView.as_view(),         name='profile'),

    # ── Phone UI transactions ─────────────────────────────────────────
    path('transactions/make/',    MakeTransactionView.as_view(),    name='make-transaction'),
    path('transactions/history/', TransactionHistoryView.as_view(), name='transaction-history'),
]
