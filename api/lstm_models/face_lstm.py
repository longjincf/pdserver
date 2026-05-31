# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
面部 LSTM 分类模型：
输入:  (batch, seq_len, 9) — [ear, lip_distance, angry, disgust, fear, happy, sad, surprise, neutral]
输出:  5 级分类 logits (0-4)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .base import BaseLSTMTrainer, SequenceDataset


# ──────────────────────────────────────────────
# 模型定义
# ──────────────────────────────────────────────
class FaceLSTM(nn.Module):
    """
    双层 LSTM → 末步隐状态 → 全连接 → 5 类 logits
    """

    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.3):
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
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        logits = self.fc(h_n[-1])
        return logits  # (batch, num_classes) — 训练时用 CrossEntropyLoss


# ──────────────────────────────────────────────
# 面部分类 Dataset（标签为 int）
# ──────────────────────────────────────────────
class FaceSequenceDataset(torch.utils.data.Dataset):
    """
    面部分类数据集，标签为 0-4 的整数。
    """

    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────
# 训练器
# ──────────────────────────────────────────────
class FaceLSTMTrainer(BaseLSTMTrainer):
    """
    data 格式:
        {
            "sequences": list[np.ndarray],  每个 shape=(seq_len, 9)
            "labels":     list[int],         0-4 分级
        }
    """

    def build_model(self, input_dim, **kwargs):
        hidden_dim = kwargs.get("hidden_dim", 64)
        num_layers = kwargs.get("num_layers", 2)
        num_classes = kwargs.get("num_classes", 5)
        dropout = kwargs.get("dropout", 0.3)
        return FaceLSTM(input_dim, hidden_dim, num_layers, num_classes, dropout)

    def build_datasets(self, data):
        sequences = data["sequences"]
        labels = data["labels"]

        indices = list(range(len(sequences)))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.config["train_test_split"], random_state=42
        )

        train_ds = FaceSequenceDataset(
            [sequences[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        val_ds = FaceSequenceDataset(
            [sequences[i] for i in val_idx],
            [labels[i] for i in val_idx],
        )
        return train_ds, val_ds

    def _get_loss_fn(self):
        return nn.CrossEntropyLoss()

    def _validate(self, val_loader, loss_fn):
        """分类任务的验证：返回 avg loss 和 accuracy（存为 mae 字段用于早停比较）"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                padded = padded.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(padded, lengths)
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * padded.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                n += padded.size(0)

        avg_loss = total_loss / n if n > 0 else float("inf")
        accuracy = correct / n if n > 0 else 0.0
        # 用 1 - accuracy 作为 "误差" 给早停逻辑（越小越好）
        return avg_loss, 1.0 - accuracy

    def predict_class(self, sequence: np.ndarray) -> int:
        """返回分类结果 0-4"""
        self.model.eval()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        length = torch.tensor([x.size(1)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x, length)
        return int(logits.argmax(dim=-1).item())

    def predict_class_batch(self, sequences: list) -> list:
        """批量分类"""
        self.model.eval()
        tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
        padded = padded.to(self.device)
        lengths = lengths.to(self.device)
        with torch.no_grad():
            logits = self.model(padded, lengths)
        return logits.argmax(dim=-1).cpu().tolist()
