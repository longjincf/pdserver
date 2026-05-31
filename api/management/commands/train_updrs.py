# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = "Train UPDRS regressor. Use --model rf (default) or --model lstm"

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='rf',
            choices=['rf', 'lstm'],
            help='Model type: rf (Random Forest, default) or lstm',
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Training epochs (LSTM only, default: 100)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size (LSTM only, default: 32)',
        )

    def handle(self, *args, **options):
        csv_path = settings.PD_DATA_PATH
        if not csv_path or not os.path.exists(csv_path):
            raise CommandError("Please set PD_DATA_PATH env var to the telemonitoring CSV path.")

        model_type = options['model']

        if model_type == 'rf':
            from api.modeling import train_and_save
            meta = train_and_save(csv_path, str(settings.MODEL_PATH))
            self.stdout.write(self.style.SUCCESS(
                f"RF model trained. Saved to {settings.MODEL_PATH}. MAE={meta['mae']:.3f}"
            ))
        else:
            from api.modeling import train_lstm_and_save
            config = {
                "epochs": options['epochs'],
                "batch_size": options['batch_size'],
            }
            result = train_lstm_and_save(csv_path, config=config)
            self.stdout.write(self.style.SUCCESS(
                f"LSTM model trained. MAE={result['mae']:.3f}, epochs={result['epochs_used']}"
            ))
