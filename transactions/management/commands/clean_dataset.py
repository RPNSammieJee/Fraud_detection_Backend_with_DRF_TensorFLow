"""
python manage.py clean_dataset
python manage.py clean_dataset --dataset 3
python manage.py clean_dataset --output media/cleaned/cleaned_transactions.csv

Pulls Transaction rows out of the DB into a DataFrame, applies the
median/mode/0 imputation rules from transactions/ml/cleaning.py, prints
a before/after missing-value report, and writes the cleaned CSV to disk
so the next phase (feature engineering, scaling, SMOTE, training) has a
stable file to read from.
"""
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from transactions.ml.cleaning import ALL_FEATURE_COLUMNS, TARGET_COLUMN, clean_dataframe
from transactions.models import Transaction


class Command(BaseCommand):
    help = 'Clean uploaded transaction data (median/mode/0 imputation) and export a cleaned CSV.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset', type=int, default=None,
            help='Only clean rows belonging to this DatasetUpload id. Default: all rows.',
        )
        parser.add_argument(
            '--output', type=str, default=None,
            help='Output CSV path. Default: media/cleaned/cleaned_transactions.csv',
        )

    def handle(self, *args, **options):
        qs = Transaction.objects.all()
        if options['dataset']:
            qs = qs.filter(dataset_id=options['dataset'])

        if not qs.exists():
            self.stderr.write(self.style.ERROR('No transactions found for that filter. Upload data first.'))
            return

        df = pd.DataFrame.from_records(qs.values(*ALL_FEATURE_COLUMNS, TARGET_COLUMN))
        self.stdout.write(f'Loaded {len(df)} rows from the database.')

        cleaned = clean_dataframe(df)

        self.stdout.write(self.style.WARNING('\nMissing values before -> after cleaning:'))
        self.stdout.write(str(cleaned.attrs['cleaning_report']))

        output_path = options['output'] or (settings.MEDIA_ROOT / 'cleaned' / 'cleaned_transactions.csv')
        output_path = str(output_path)
        cleaned.to_csv(output_path, index=False)

        self.stdout.write(self.style.SUCCESS(f'\nWrote cleaned dataset to {output_path} ({len(cleaned)} rows).'))
