# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
LSTM 模型训练基础设施：
- 通用训练循环（含早停、学习率调度）
- 序列数据 collate（padding + masking）
- 模型保存 / 加载
- 推理接口
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from copy import deepcopy


# ──────────────────────────────────────────────
# 默认超参
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "train_test_split": 0.2,
    "optimizer": "Adam",
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
}


# ──────────────────────────────────────────────
# 通用序列 Dataset
# ──────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    每条样本是一个变长序列 + 一个标签。
    sequences: list[np.ndarray]  每个 shape = (seq_len, feature_dim)
    labels:    list[float] | list[int]
    """

    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_variable_length(batch):
    """
    将变长序列 padding 到 batch 内最大长度，返回 (padded, lengths, labels)。
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded, lengths, labels


# ──────────────────────────────────────────────
# 训练器基类
# ──────────────────────────────────────────────
class BaseLSTMTrainer:
    """
    子类需要实现:
        - build_model(input_dim, **kwargs) -> nn.Module
        - build_datasets(data) -> (train_dataset, val_dataset)
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_state = None
        self.input_dim = None

    # ── 子类必须实现 ──
    def build_model(self, input_dim, **kwargs):
        raise NotImplementedError

    def build_datasets(self, data):
        """返回 (train_dataset: SequenceDataset, val_dataset: SequenceDataset)"""
        raise NotImplementedError

    # ── 训练循环 ──
    def train(self, data, save_path, **model_kwargs):
        """
        data: 原始数据（格式由子类解释）
        save_path: 模型保存路径（.pt）
        返回: {"mae": float, "epochs_used": int}
        """
        train_ds, val_ds = self.build_datasets(data)
        self.input_dim = train_ds[0][0].shape[-1]

        self.model = self.build_model(self.input_dim, **model_kwargs).to(self.device)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_variable_length,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_variable_length,
        )

        optimizer = getattr(torch.optim, self.config["optimizer"])(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.config["scheduler_factor"],
            patience=self.config["scheduler_patience"],
        )

        # 回归用 MSELoss，分类由子类重写 loss_fn
        loss_fn = self._get_loss_fn()
        best_val_loss = float("inf")
        patience_counter = 0
        epochs_used = 0

        for epoch in range(1, self.config["epochs"] + 1):
            # ── train ──
            self.model.train()
            train_loss = 0.0
            for padded, lengths, labels in train_loader:
                padded = padded.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(padded, lengths)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * padded.size(0)

            train_loss /= len(train_ds)

            # ── validate ──
            val_loss, val_mae = self._validate(val_loader, loss_fn)
            scheduler.step(val_loss)
            epochs_used = epoch

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    break

        # 恢复最优权重并保存
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self._save(save_path, val_mae)

        return {"mae": val_mae, "epochs_used": epochs_used}

    def _validate(self, val_loader, loss_fn):
        self.model.eval()
        total_loss = 0.0
        total_abs = 0.0
        n = 0
        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                padded = padded.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                output = self.model(padded, lengths)
                loss = loss_fn(output, labels)
                total_loss += loss.item() * padded.size(0)
                total_abs += torch.abs(output - labels).sum().item()
                n += padded.size(0)

        avg_loss = total_loss / n if n > 0 else float("inf")
        mae = total_abs / n if n > 0 else float("inf")
        return avg_loss, mae

    def _get_loss_fn(self):
        """子类可重写，例如分类用 CrossEntropyLoss"""
        return nn.MSELoss()

    # ── 保存 / 加载 ──
    def _save(self, save_path, val_mae):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.best_state or self.model.state_dict(),
                "config": self.config,
                "input_dim": self.input_dim,
                "val_mae": val_mae,
            },
            str(save_path),
        )

    def load(self, model_path):
        """加载已训练模型，返回 self.model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.config = {**DEFAULT_CONFIG, **checkpoint.get("config", {})}
        self.model = self.build_model(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return self.model

    # ── 推理 ──
    def predict(self, sequence: np.ndarray) -> float:
        """
        输入: sequence, shape = (seq_len, feature_dim)
        输出: float（回归值 or 分类 logits 取 argmax）
        """
        self.model.eval()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        length = torch.tensor([x.size(1)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model(x, length)
        return output.cpu().item()

    def predict_batch(self, sequences: list) -> list:
        """
        输入: list[np.ndarray]，每个 shape = (seq_len, feature_dim)
        输出: list[float]
        """
        self.model.eval()
        tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
        padded = padded.to(self.device)
        lengths = lengths.to(self.device)
        with torch.no_grad():
            output = self.model(padded, lengths)
        return output.cpu().tolist()
