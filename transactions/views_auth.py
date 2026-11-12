"""
Auth + transaction views for the phone UI.
Kept separate from views.py so existing dataset/predict endpoints are untouched.
"""
from datetime import timedelta

from django.contrib.auth import authenticate
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import TransactionRecord, UserProfile
from .ml.predictor import ModelNotReadyError, predict_single
from .serializers_auth import (
    LoginSerializer,
    MakeTransactionSerializer,
    SignupSerializer,
    TransactionRecordSerializer,
    UserProfileSerializer,
)


def _tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access':  str(refresh.access_token),
    }


class SignupView(APIView):
    """POST /api/auth/signup/"""
    permission_classes = []

    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {'detail': 'Account created successfully.', 'tokens': _tokens_for_user(user)},
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    """POST /api/auth/login/"""
    permission_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        return Response(
            {'detail': 'Login successful.', 'tokens': _tokens_for_user(user)},
            status=status.HTTP_200_OK,
        )


class ProfileView(APIView):
    """GET /api/auth/profile/"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response({'detail': 'Profile not found.'}, status=status.HTTP_404_NOT_FOUND)
        return Response(UserProfileSerializer(profile).data)


class MakeTransactionView(APIView):
    """
    POST /api/transactions/make/

    Accepts the phone-UI transaction form, auto-derives all ML features,
    runs the fraud predictor, saves the result, and returns it.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = MakeTransactionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Verify password confirmation
        user = request.user
        if not authenticate(username=user.username, password=data['password']):
            return Response({'detail': 'Incorrect password.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            profile = user.profile
        except UserProfile.DoesNotExist:
            return Response({'detail': 'User profile not found.'}, status=status.HTTP_404_NOT_FOUND)

        # ── Auto-derive ML features ──────────────────────────────────
        now = timezone.localtime(timezone.now())
        transaction_hour = now.hour

        foreign_transaction = (
            0 if data['transaction_country'].strip().lower() == profile.country.strip().lower()
            else 1
        )
        location_mismatch = (
            0 if data['transaction_location'].strip().lower() == profile.location.strip().lower()
            else 1
        )
        device_trust_score = 85.0 if data['is_personal_device'] else 25.0

        # Count transactions this user made in the last 24 hours
        since = now - timedelta(hours=24)
        velocity_last_24h = float(
            TransactionRecord.objects.filter(user=user, created_at__gte=since).count()
        )

        cardholder_age = float(profile.age)

        # ── Build prediction payload ─────────────────────────────────
        prediction_input = {
            'amount':               data['amount'],
            'transaction_hour':     transaction_hour,
            'merchant_category':    data['merchant_category'],
            'foreign_transaction':  foreign_transaction,
            'location_mismatch':    location_mismatch,
            'device_trust_score':   device_trust_score,
            'velocity_last_24h':    velocity_last_24h,
            'cardholder_age':       cardholder_age,
        }

        try:
            prediction = predict_single(prediction_input)
        except ModelNotReadyError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # ── Save transaction record ──────────────────────────────────
        record = TransactionRecord.objects.create(
            user=user,
            amount=data['amount'],
            merchant_category=data['merchant_category'],
            recipient_account=data.get('recipient_account', ''),
            transaction_country=data['transaction_country'],
            transaction_location=data['transaction_location'],
            is_personal_device=data['is_personal_device'],
            transaction_hour=transaction_hour,
            foreign_transaction=foreign_transaction,
            location_mismatch=location_mismatch,
            device_trust_score=device_trust_score,
            velocity_last_24h=velocity_last_24h,
            cardholder_age=cardholder_age,
            fraud_probability=prediction['fraud_probability'],
            is_fraud=prediction['is_fraud'],
            threshold_used=prediction['threshold_used'],
            status=TransactionRecord.Status.FLAGGED if prediction['is_fraud'] else TransactionRecord.Status.COMPLETED,
        )

        profile.balance = max(0, float(profile.balance) - data['amount'])
        profile.save(update_fields=['balance'])

        return Response({
            'transaction_id':    record.id,
            'prediction':        prediction,
            'new_balance':      float(profile.balance),
            'derived_features': {
                'transaction_hour':   transaction_hour,
                'foreign_transaction': foreign_transaction,
                'location_mismatch':  location_mismatch,
                'device_trust_score': device_trust_score,
                'velocity_last_24h':  velocity_last_24h,
                'cardholder_age':     cardholder_age,
            },
        }, status=status.HTTP_201_CREATED)


class TransactionHistoryView(generics.ListAPIView):
    """GET /api/transactions/history/"""
    permission_classes = [IsAuthenticated]
    serializer_class   = TransactionRecordSerializer

    def get_queryset(self):
        return TransactionRecord.objects.filter(user=self.request.user)
