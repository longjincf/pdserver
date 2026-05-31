# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Train face severity classifier using LSTM. Requires a CSV with 'sequence' and 'grade' columns."

    def add_arguments(self, parser):
        parser.add_argument(
            '--data',
            type=str,
            required=True,
            help='Path to training CSV (columns: sequence, grade)',
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Training epochs (default: 100)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size (default: 32)',
        )

    def handle(self, *args, **options):
        csv_path = options['data']
        if not os.path.exists(csv_path):
            raise CommandError(f"File not found: {csv_path}")

        import json
        import numpy as np
        import pandas as pd
        from api.face_modeling import train_face_lstm

        df = pd.read_csv(csv_path)

        sequences = []
        labels = []

        if "sequence" not in df.columns:
            raise CommandError("CSV must have a 'sequence' column with JSON arrays.")
        if "grade" not in df.columns:
            raise CommandError("CSV must have a 'grade' column with 0-4 integer labels.")

        for _, row in df.iterrows():
            seq = json.loads(row["sequence"]) if isinstance(row["sequence"], str) else row["sequence"]
            sequences.append(np.array(seq, dtype=np.float32))
            labels.append(int(row["grade"]))

        config = {
            "epochs": options['epochs'],
            "batch_size": options['batch_size'],
        }

        result = train_face_lstm(
            {"sequences": sequences, "labels": labels},
            config=config,
        )
        self.stdout.write(self.style.SUCCESS(
            f"Face LSTM trained. Error={result['mae']:.4f}, epochs={result['epochs_used']}"
        ))
