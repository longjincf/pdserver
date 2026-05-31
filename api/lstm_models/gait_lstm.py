# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
步态 LSTM 回归模型：
输入:  (batch, seq_len, 5) — [left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, hip_y]
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
class GaitLSTM(nn.Module):
    """
    双层 LSTM → 取末步隐状态 → 全连接 → UPDRS 回归
    """

    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, dropout=0.3):
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
        # x: (batch, max_len, input_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])  # 取最后一层的隐状态
        return out.squeeze(-1)


# ──────────────────────────────────────────────
# 训练器
# ──────────────────────────────────────────────
class GaitLSTMTrainer(BaseLSTMTrainer):
    """
    data 格式（由调用方准备）:
        {
            "sequences": list[np.ndarray],  每个 shape=(seq_len, 5)
            "labels":     list[float],       gait_score (0–4) 或 UPDRS
        }
    """

    def build_model(self, input_dim, **kwargs):
        hidden_dim = kwargs.get("hidden_dim", 128)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.3)
        return GaitLSTM(input_dim, hidden_dim, num_layers, dropout)

    def build_datasets(self, data):
        sequences = data["sequences"]
        labels = data["labels"]

        # 划分训练 / 验证集
        indices = list(range(len(sequences)))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.config["train_test_split"], random_state=42
        )

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = SequenceDataset(train_seqs, train_labels)
        val_ds = SequenceDataset(val_seqs, val_labels)
        return train_ds, val_ds
