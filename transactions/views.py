import pandas as pd
from django.db import transaction as db_transaction
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .ml.predictor import ModelNotReadyError, predict_single
from .serializers import PredictionInputSerializer

from .models import DatasetUpload, Transaction
from .serializers import (
    DatasetUploadSerializer,
    TransactionSerializer,
    PredictionInputSerializer,
)

# Must match the columns in the credit-card-fraud CSV exactly.
EXPECTED_COLUMNS = [
    'amount', 'transaction_hour', 'merchant_category', 'foreign_transaction',
    'location_mismatch', 'device_trust_score', 'velocity_last_24h',
    'cardholder_age', 'is_fraud',
]


class DatasetUploadView(APIView):
    """
    POST /api/datasets/upload/
    multipart/form-data, field name "file" = the .csv

    Parses the CSV with pandas and bulk-creates one Transaction row per
    record, linked back to the DatasetUpload. Missing values are kept as
    NULL in the DB on purpose — cleaning happens in a separate step
    (see the `clean_dataset` management command) so the raw upload stays
    an honest record of what was actually given to us.
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response(
                {'detail': 'No file provided. Use multipart form field "file".'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not file_obj.name.lower().endswith('.csv'):
            return Response({'detail': 'Only .csv files are supported.'}, status=status.HTTP_400_BAD_REQUEST)

        dataset = DatasetUpload.objects.create(file=file_obj, status=DatasetUpload.Status.PROCESSING)

        try:
            df = pd.read_csv(dataset.file.path)
        except Exception as exc:  # noqa: BLE001 - want to report any parse failure to the user
            dataset.status = DatasetUpload.Status.FAILED
            dataset.error_message = f'Could not parse CSV: {exc}'
            dataset.save()
            return Response({'detail': dataset.error_message}, status=status.HTTP_400_BAD_REQUEST)

        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            dataset.status = DatasetUpload.Status.FAILED
            dataset.error_message = f'Missing expected columns: {sorted(missing_cols)}'
            dataset.save()
            return Response({'detail': dataset.error_message}, status=status.HTTP_400_BAD_REQUEST)

        has_id_col = 'transaction_id' in df.columns
        cols_to_keep = EXPECTED_COLUMNS + (['transaction_id'] if has_id_col else [])
        df = df[cols_to_keep].where(pd.notnull(df[cols_to_keep]), None)

        objs = [
            Transaction(
                dataset=dataset,
                source_transaction_id=str(row['transaction_id']) if has_id_col and row['transaction_id'] is not None else None,
                amount=row['amount'],
                transaction_hour=row['transaction_hour'],
                merchant_category=row['merchant_category'],
                foreign_transaction=row['foreign_transaction'],
                location_mismatch=row['location_mismatch'],
                device_trust_score=row['device_trust_score'],
                velocity_last_24h=row['velocity_last_24h'],
                cardholder_age=row['cardholder_age'],
                is_fraud=int(row['is_fraud']) if row['is_fraud'] is not None else 0,
            )
            for row in df.to_dict(orient='records')
        ]

        with db_transaction.atomic():
            Transaction.objects.bulk_create(objs, batch_size=1000)

        dataset.status = DatasetUpload.Status.COMPLETED
        dataset.row_count = len(objs)
        dataset.save()

        return Response(DatasetUploadSerializer(dataset).data, status=status.HTTP_201_CREATED)


class DatasetListView(generics.ListAPIView):
    """GET /api/datasets/ - history of uploads, newest first."""
    queryset = DatasetUpload.objects.all()
    serializer_class = DatasetUploadSerializer


class TransactionListView(generics.ListAPIView):
    """
    GET /api/transactions/
    GET /api/transactions/?dataset=<id>  - restrict to one upload
    """
    serializer_class = TransactionSerializer

    def get_queryset(self):
        qs = Transaction.objects.all()
        dataset_id = self.request.query_params.get('dataset')
        if dataset_id:
            qs = qs.filter(dataset_id=dataset_id)
        return qs

class PredictView(APIView):
    """
    POST /api/predict/
    JSON body matching PredictionInputSerializer, e.g.:
    {
      "amount": 84.47,
      "transaction_hour": 22,
      "merchant_category": "Electronics",
      "foreign_transaction": 0,
      "location_mismatch": 0,
      "device_trust_score": 66,
      "velocity_last_24h": 3,
      "cardholder_age": 40
    }

    Runs the submitted transaction through the exact same cleaning +
    feature engineering + scaling pipeline used at training time, then
    the trained model, and returns a fraud probability + label.
    """

    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = predict_single(serializer.validated_data)
        except ModelNotReadyError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response(result, status=status.HTTP_200_OK)

