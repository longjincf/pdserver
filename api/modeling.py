# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from django.conf import settings

# Columns we will use (must exist in the telemonitoring dataset)
CANDIDATE_COLUMNS = [
    # Common acoustic measures in telemonitoring dataset:
    'Jitter(%)', 'Shimmer', 'NHR', 'HNR',
    # Plus a few extras if present (model is robust to missing ones)
    'Jitter:DDP', 'Shimmer(dB)', 'RPDE', 'DFA', 'PPE', 'APQ5', 'APQ', 'DDA'
]

MODEL_PATH = settings.MODEL_PATH          # model/model.pkl
LSTM_MODEL_PATH = settings.MODEL_DIR / "voice_lstm.pt"


# ──────────────────────────────────────────────
# Random Forest（原有逻辑，保留兼容）
# ──────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def make_train_data(df: pd.DataFrame):
    # Keep only available columns
    cols = [c for c in CANDIDATE_COLUMNS if c in df.columns]
    X = df[cols].copy()
    y = df['total_UPDRS'] if 'total_UPDRS' in df.columns else df['motor_UPDRS']
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols

def train_and_save(csv_path: str, model_path: str):
    df = load_dataset(csv_path)
    X, y, cols = make_train_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    meta = {"feature_columns": cols, "mae": float(mae)}
    dump({"model": pipe, "meta": meta}, model_path)
    return meta

def load_model(model_path: str):
    return load(model_path)

def predict_updrs_from_features(model_obj, feats: dict) -> float:
    model = model_obj['model']
    meta = model_obj['meta']
    cols = meta['feature_columns']
    # Build a single-row dataframe with available columns, fill missing with zeros
    row = {c: 0.0 for c in cols}
    # Map incoming features
    mapping = {
        'Jitter(%)': 'jitter_local',
        'Shimmer': 'shimmer_local',
        'HNR': 'hnr_db',
        'NHR': 'nhr', # not available from parselmouth block; will remain 0 unless provided
    }
    for col, src in mapping.items():
        if col in row and src in feats:
            row[col] = feats[src]
    X = pd.DataFrame([row])
    updrs = float(model.predict(X)[0])
    return updrs

def updrs_to_bucket(updrs: float) -> int:
    # Example cutoffs — adjust with clinical guidance or empirical quantiles later.
    # 0–15 -> 0, 15–30 -> 1, 30–45 -> 2, 45–60 -> 3, >60 -> 4
    if updrs <= 15: return 0
    if updrs <= 30: return 1
    if updrs <= 45: return 2
    if updrs <= 60: return 3
    return 4


# ──────────────────────────────────────────────
# LSTM（新增）
# ──────────────────────────────────────────────
def train_lstm_and_save(csv_path: str, config=None, **model_kwargs):
    """
    从 CSV 训练语音 LSTM 模型。
    CSV 格式要求:
        - 每行一个样本
        - 列: sequence (JSON 字符串), total_UPDRS
        - 或者兼容旧格式（聚合特征），自动构造伪时序

    如果 CSV 中有 "sequence" 列，则解析为时序；
    否则回退到用聚合特征构造伪时序。
    """
    from .lstm_models.voice_lstm import VoiceLSTMTrainer

    df = pd.read_csv(csv_path)

    sequences = []
    labels = []

    if "sequence" in df.columns:
        import json
        for _, row in df.iterrows():
            seq = json.loads(row["sequence"]) if isinstance(row["sequence"], str) else row["sequence"]
            sequences.append(np.array(seq, dtype=np.float32))
            label_col = "total_UPDRS" if "total_UPDRS" in df.columns else "motor_UPDRS"
            labels.append(float(row[label_col]))
    else:
        # 兼容旧格式：从聚合特征构造伪时序
        cols = [c for c in CANDIDATE_COLUMNS if c in df.columns]
        label_col = "total_UPDRS" if "total_UPDRS" in df.columns else "motor_UPDRS"

        for _, row in df.iterrows():
            vals = [float(row[c]) for c in cols if pd.notna(row.get(c))]
            # 构造 (len(vals), 3) 的伪序列：将特征循环映射到 3 维
            seq = np.zeros((max(len(vals), 1), 3), dtype=np.float32)
            for i, v in enumerate(vals):
                seq[i, i % 3] = v
            sequences.append(seq)
            labels.append(float(row[label_col]))

    data = {"sequences": sequences, "labels": labels}
    trainer = VoiceLSTMTrainer(config)
    result = trainer.train(data, str(LSTM_MODEL_PATH), **model_kwargs)
    return result


def load_voice_model():
    """
    优先加载 LSTM 模型，fallback 到 RF 模型。
    返回:
        {"type": "lstm", "model": VoiceLSTMTrainer, "meta": {...}}
        {"type": "rf",   "model": <joblib obj>,     "meta": {...}}
    或 None
    """
    if LSTM_MODEL_PATH.exists():
        try:
            from .lstm_models.voice_lstm import VoiceLSTMTrainer
            trainer = VoiceLSTMTrainer()
            trainer.load(str(LSTM_MODEL_PATH))
            import torch
            checkpoint = torch.load(
                str(LSTM_MODEL_PATH), map_location="cpu", weights_only=False
            )
            return {
                "type": "lstm",
                "model": trainer,
                "meta": {"val_mae": checkpoint.get("val_mae"), "input_dim": checkpoint.get("input_dim")},
            }
        except Exception:
            pass  # fallback

    if Path(MODEL_PATH).exists():
        obj = load(MODEL_PATH)
        return {"type": "rf", "model": obj, "meta": obj.get("meta", {})}

    return None


def predict_voice(model_obj, feats_or_sequence):
    """
    统一预测入口：根据模型类型自动选择 RF 或 LSTM。
    feats_or_sequence:
        - dict with key "sequence" (np.ndarray) → 直接用时序
        - dict with voice features (jitter_local, shimmer_local, hnr_db) → RF 或构造伪时序
        - np.ndarray → 直接用时序
    """
    if model_obj["type"] == "lstm":
        trainer = model_obj["model"]
        if isinstance(feats_or_sequence, dict) and "sequence" in feats_or_sequence:
            seq = np.array(feats_or_sequence["sequence"], dtype=np.float32)
        elif isinstance(feats_or_sequence, np.ndarray):
            seq = feats_or_sequence
        else:
            # 从聚合特征构造伪时序
            vals = [feats_or_sequence.get("jitter_local", 0),
                    feats_or_sequence.get("shimmer_local", 0),
                    feats_or_sequence.get("hnr_db", 0)]
            seq = np.array([vals], dtype=np.float32)
        return trainer.predict(seq)
    else:
        return predict_updrs_from_features(model_obj["model"], feats_or_sequence)
