# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
语音 LSTM 回归模型：
输入:  (batch, seq_len, 3) — [jitter_local, shimmer_local, hnr_db] 逐帧
输出:  UPDRS 回归值
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .base import BaseLSTMTrainer, SequenceDataset


# ──────────────────────────────────────────────
# 模型定义
# ──────────────────────────────────────────────
class VoiceLSTM(nn.Module):
    """
    双层 LSTM → 末步隐状态 → 全连接 → UPDRS 回归
    """

    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


# ──────────────────────────────────────────────
# 训练器
# ──────────────────────────────────────────────
class VoiceLSTMTrainer(BaseLSTMTrainer):
    """
    data 格式:
        {
            "sequences": list[np.ndarray],  每个 shape=(seq_len, 3)
            "labels":     list[float],       total_UPDRS
        }
    """

    def build_model(self, input_dim, **kwargs):
        hidden_dim = kwargs.get("hidden_dim", 128)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.3)
        return VoiceLSTM(input_dim, hidden_dim, num_layers, dropout)

    def build_datasets(self, data):
        sequences = data["sequences"]
        labels = data["labels"]

        indices = list(range(len(sequences)))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.config["train_test_split"], random_state=42
        )

        train_ds = SequenceDataset(
            [sequences[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        val_ds = SequenceDataset(
            [sequences[i] for i in val_idx],
            [labels[i] for i in val_idx],
        )
        return train_ds, val_ds
