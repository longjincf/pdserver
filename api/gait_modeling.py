# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

MODEL_PATH = settings.MODEL_DIR / "gait_model.pkl"
LSTM_MODEL_PATH = settings.MODEL_DIR / "gait_lstm.pt"


# ──────────────────────────────────────────────
# Random Forest（原有逻辑，保留兼容）
# ──────────────────────────────────────────────
def load_gait_csv(csv_path):
    return pd.read_csv(csv_path)


def make_train_data(df: pd.DataFrame):
    """
    从生成的 CSV 中提取特征列和标签列
    - 特征列: 所有以 'col_' 开头的列
    - 标签列: gait_score
    """
    cols = [c for c in df.columns if c.startswith("col_")]
    if not cols:
        raise ValueError("CSV 中未找到任何 gait 特征列，请检查 CSV 文件格式。")

    if "gait_score" not in df.columns:
        raise ValueError("CSV 中未找到 gait_score 列，请检查 CSV 文件。")
    y = df["gait_score"]

    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    return X, y, cols


def train_and_save(csv_path: str):
    df = load_gait_csv(csv_path)
    X, y, cols = make_train_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    meta = {
        "feature_columns": cols,
        "mae": float(mae)
    }

    dump({"model": pipe, "meta": meta}, MODEL_PATH)

    return meta


def predict_from_features(model_obj, feats: dict) -> float:
    """
    输入: feats 是一个 dict, 例如:
    {
        'col_0_mean': 61.1,
        'col_0_std': 88.7,
        ...
    }
    输出: 预测的 gait_score
    """
    model = model_obj["model"]
    cols = model_obj["meta"]["feature_columns"]

    row = {c: 0.0 for c in cols}
    for k, v in feats.items():
        if k in row:
            row[k] = float(v)
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])


def updrs_to_bucket(updrs: float) -> int:
    """将 gait_score 转换为 0-4 的分级"""
    if updrs <= 0.5:
        return 0
    if updrs <= 1.5:
        return 1
    if updrs <= 2.5:
        return 2
    if updrs <= 3.5:
        return 3
    return 4


# ──────────────────────────────────────────────
# LSTM（新增）
# ──────────────────────────────────────────────
def train_lstm_and_save(csv_path: str, config=None, **model_kwargs):
    """
    从 CSV 训练步态 LSTM 模型。
    CSV 格式要求:
        - 每行一个样本
        - 列: sequence (JSON 字符串或分号分隔的浮点数), gait_score
        - 或者列: frames_0 ... frames_N (每帧 5 维展开), gait_score

    如果 CSV 中有 "sequence" 列，则解析为时序；
    否则回退到用聚合特征构造伪时序。
    """
    from .lstm_models.gait_lstm import GaitLSTMTrainer

    df = pd.read_csv(csv_path)

    sequences = []
    labels = []

    if "sequence" in df.columns:
        # 列中存储 JSON 格式的时序数据
        import json
        for _, row in df.iterrows():
            seq = json.loads(row["sequence"]) if isinstance(row["sequence"], str) else row["sequence"]
            sequences.append(np.array(seq, dtype=np.float32))
            labels.append(float(row["gait_score"]))
    else:
        # 从聚合特征构造伪时序（兼容旧格式 CSV）
        # 用每个 col_ 的 mean 和 std 模拟一条高斯时序
        col_cols = [c for c in df.columns if c.startswith("col_")]
        if "gait_score" not in df.columns:
            raise ValueError("CSV 中未找到 gait_score 列，请检查 CSV 文件。")

        for _, row in df.iterrows():
            vals = [row[c] for c in col_cols if pd.notna(row[c])]
            # 构造 (len(vals), 1) 的伪序列，复制到 5 维
            seq = np.column_stack([vals] * 5).astype(np.float32)
            sequences.append(seq)
            labels.append(float(row["gait_score"]))

    data = {"sequences": sequences, "labels": labels}
    trainer = GaitLSTMTrainer(config)
    result = trainer.train(data, str(LSTM_MODEL_PATH), **model_kwargs)
    return result


def load_model():
    """
    优先加载 LSTM 模型，fallback 到 RF 模型。
    返回 dict:
        {"type": "lstm", "model": <LSTM trainer>, "meta": {...}}
        {"type": "rf",   "model": <joblib obj>,   "meta": {...}}
    或 None（无可用模型时）
    """
    if LSTM_MODEL_PATH.exists():
        try:
            from .lstm_models.gait_lstm import GaitLSTMTrainer
            trainer = GaitLSTMTrainer()
            trainer.load(str(LSTM_MODEL_PATH))
            checkpoint = __import__("torch").load(
                str(LSTM_MODEL_PATH), map_location="cpu", weights_only=False
            )
            return {
                "type": "lstm",
                "model": trainer,
                "meta": {"val_mae": checkpoint.get("val_mae"), "input_dim": checkpoint.get("input_dim")},
            }
        except Exception:
            pass  # fallback

    if MODEL_PATH.exists():
        obj = load(MODEL_PATH)
        return {"type": "rf", "model": obj, "meta": obj.get("meta", {})}

    return None


def predict_lstm(model_obj, feats_or_sequence):
    """
    使用 LSTM 模型预测。
    feats_or_sequence:
        - dict with key "sequence" (np.ndarray) → 直接用时序
        - dict with RF-style features → 兼容旧接口
    """
    trainer = model_obj["model"]
    if isinstance(feats_or_sequence, dict) and "sequence" in feats_or_sequence:
        seq = np.array(feats_or_sequence["sequence"], dtype=np.float32)
    elif isinstance(feats_or_sequence, np.ndarray):
        seq = feats_or_sequence
    else:
        # 旧接口兼容：从聚合特征构造伪时序
        vals = [v for v in feats_or_sequence.values() if isinstance(v, (int, float))]
        seq = np.column_stack([vals] * 5).astype(np.float32)

    return trainer.predict(seq)


def predict_gait(model_obj, feats_or_sequence):
    """
    统一预测入口：根据模型类型自动选择 RF 或 LSTM。
    """
    if model_obj["type"] == "lstm":
        return predict_lstm(model_obj, feats_or_sequence)
    else:
        return predict_from_features(model_obj, feats_or_sequence)
