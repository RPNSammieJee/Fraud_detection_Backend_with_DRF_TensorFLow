
"""
Serializers for signup, login, user profile, and transaction records.
Kept in a separate file so the existing serializers.py is untouched.
"""
from datetime import date, timedelta, timezone
import datetime

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone as dj_timezone
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import UserProfile, TransactionRecord


class SignupSerializer(serializers.Serializer):
    # Django User fields
    full_name       = serializers.CharField(max_length=150)
    username        = serializers.CharField(max_length=150)
    email           = serializers.EmailField()
    password        = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)

    # Profile fields
    phone_number    = serializers.CharField(max_length=20, required=False, allow_blank=True)
    date_of_birth   = serializers.DateField()
    country         = serializers.CharField(max_length=100)
    location        = serializers.CharField(max_length=200)

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError('Username already taken.')
        return value

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError('Email already registered.')
        return value

    def validate_date_of_birth(self, value):
        today = date.today()
        age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
        if age < 18:
            raise serializers.ValidationError('You must be at least 18 years old.')
        return value

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError({'confirm_password': 'Passwords do not match.'})
        validate_password(data['password'])
        return data

    def create(self, validated_data):
        full_name = validated_data.pop('full_name')
        validated_data.pop('confirm_password')
        phone_number  = validated_data.pop('phone_number', '')
        date_of_birth = validated_data.pop('date_of_birth')
        country       = validated_data.pop('country')
        location      = validated_data.pop('location')

        first, *rest = full_name.strip().split(' ', 1)
        last = rest[0] if rest else ''

        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=first,
            last_name=last,
        )
        UserProfile.objects.create(
            user=user,
            phone_number=phone_number,
            date_of_birth=date_of_birth,
            country=country,
            location=location,
        )
        return user


class LoginSerializer(serializers.Serializer):
    identifier = serializers.CharField(help_text='Username or email')
    password   = serializers.CharField(write_only=True)

    def validate(self, data):
        from django.contrib.auth import authenticate

        identifier = data['identifier']
        password   = data['password']

        # Allow login by email OR username
        user = None
        if '@' in identifier:
            try:
                u = User.objects.get(email=identifier)
                user = authenticate(username=u.username, password=password)
            except User.DoesNotExist:
                pass
        else:
            user = authenticate(username=identifier, password=password)

        if not user:
            raise serializers.ValidationError('Invalid credentials.')
        if not user.is_active:
            raise serializers.ValidationError('Account is disabled.')

        data['user'] = user
        return data


class UserProfileSerializer(serializers.ModelSerializer):
    full_name      = serializers.SerializerMethodField()
    username       = serializers.CharField(source='user.username', read_only=True)
    email          = serializers.EmailField(source='user.email', read_only=True)
    age            = serializers.IntegerField(read_only=True)

    class Meta:
        model  = UserProfile
        fields = [
            'full_name', 'username', 'email', 'phone_number',
            'date_of_birth', 'country', 'location',
            'account_number', 'credit_card_number', 'age',
            'balance', 'created_at', 'updated_at',
        ]
        read_only_fields = ['account_number', 'credit_card_number', 'created_at', 'updated_at']

    def get_full_name(self, obj):
        return f'{obj.user.first_name} {obj.user.last_name}'.strip()


class TransactionRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model  = TransactionRecord
        fields = '__all__'
        read_only_fields = [
            'user', 'transaction_hour', 'foreign_transaction',
            'location_mismatch', 'device_trust_score', 'velocity_last_24h',
            'cardholder_age', 'fraud_probability', 'is_fraud',
            'threshold_used', 'status', 'created_at',
        ]


class MakeTransactionSerializer(serializers.Serializer):
    """Input from the phone UI transaction form."""
    amount               = serializers.FloatField(min_value=0.01)
    merchant_category    = serializers.ChoiceField(
        choices=['Electronics', 'Travel', 'Grocery', 'Food', 'Clothing'])
    recipient_account    = serializers.CharField(max_length=64, required=False, allow_blank=True)
    transaction_country  = serializers.CharField(max_length=100)
    transaction_location = serializers.CharField(max_length=200)
    is_personal_device   = serializers.BooleanField()
    password             = serializers.CharField(write_only=True,
                           help_text='User confirms with their login password.')
