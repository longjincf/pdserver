# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
面部评估模型管理：
- LSTM 分类模型（0-4 级）
- fallback 到规则打分
"""

from pathlib import Path
from django.conf import settings

FACE_LSTM_MODEL_PATH = settings.MODEL_DIR / "face_lstm.pt"

# 症状描述映射（与 feature_extractor_face.py 中的 desc_map 保持一致）
GRADE_DESCRIPTIONS = [
    "正常：正常的面部表情",
    "轻微：仅有瞬目频率的减少",
    "轻度：除瞬目频率减少外，下面部表情也减少",
    "中度：面部表情减少，且嘴唇在不说话时偶尔张开",
    "重度：面部表情显著减少，且嘴唇在不说话时经常张开",
]

GRADE_LABELS = ["正常", "轻微", "轻度", "中度", "重度"]


def load_face_lstm_model():
    """
    加载面部 LSTM 模型。
    返回:
        {"type": "lstm", "model": FaceLSTMTrainer, "meta": {...}}
    或 None
    """
    if not FACE_LSTM_MODEL_PATH.exists():
        return None
    try:
        from .lstm_models.face_lstm import FaceLSTMTrainer
        trainer = FaceLSTMTrainer()
        trainer.load(str(FACE_LSTM_MODEL_PATH))
        import torch
        checkpoint = torch.load(
            str(FACE_LSTM_MODEL_PATH), map_location="cpu", weights_only=False
        )
        return {
            "type": "lstm",
            "model": trainer,
            "meta": {"val_error": checkpoint.get("val_mae"), "input_dim": checkpoint.get("input_dim")},
        }
    except Exception:
        return None


def predict_face_grade(model_obj, sequence):
    """
    使用 LSTM 模型预测面部分级。
    sequence: np.ndarray, shape=(N, 9)
    返回: (grade: int, description: str)
    """
    trainer = model_obj["model"]
    grade = trainer.predict_class(sequence)
    grade = max(0, min(4, grade))
    description = f"{GRADE_LABELS[grade]}：{GRADE_DESCRIPTIONS[grade]}"
    return int(grade), description


def train_face_lstm(data, config=None, **model_kwargs):
    """
    训练面部 LSTM 分类模型。
    data 格式:
        {
            "sequences": list[np.ndarray],  每个 shape=(seq_len, 9)
            "labels":     list[int],         0-4
        }
    """
    from .lstm_models.face_lstm import FaceLSTMTrainer

    trainer = FaceLSTMTrainer(config)
    result = trainer.train(data, str(FACE_LSTM_MODEL_PATH), **model_kwargs)
    return result
