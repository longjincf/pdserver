# api/feature_extractor_face.py
# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
from pathlib import Path
import numpy as np, pandas as pd, cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from onnxruntime import InferenceSession

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BUFFALO_L = MODEL_DIR / "buffalo_l"
EMOTION_ONNX = MODEL_DIR / "mini_xception.onnx"
IMAGE_SIZE = 112

app = FaceAnalysis(name=str(BUFFALO_L), providers=["CPUExecutionProvider"],
                   allowed_modules=["detection", "landmark_2d_106"])
app.prepare(ctx_id=0, det_size=(640, 640))

fer_sess = InferenceSession(str(EMOTION_ONNX), providers=["CPUExecutionProvider"])
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

LEFT_EYE  = [35, 36, 33, 37, 39, 42, 40, 41]
RIGHT_EYE = [89, 90, 87, 88, 91, 94, 92, 93]
UP_LIP, LO_LIP = 66, 62

def predict_emotion(gray112: np.ndarray) -> str:
    gray64 = cv2.resize(gray112, (64, 64))
    blob = gray64.astype(np.float32)[None, :, :, None] / 255.0
    logits = fer_sess.run(None, {"input": blob})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    return EMOTION_LABELS[int(np.argmax(probs))]

def ear(pts: np.ndarray) -> float:
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

class OfflineAnalyzer:
    def __init__(self, video_path, target_fps=3):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(video_path)
        self.target_fps = target_fps
        self.real_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.skip = max(1, int(self.real_fps / target_fps))
        self.duration = max(1.0, self.frames / self.real_fps)

    def run(self):
        ears, lips, emotions = [], [], []
        for idx in range(0, self.frames, self.skip):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok: continue
            faces = app.get(frame)
            if not faces: continue
            face = faces[0]
            kps = face.landmark_2d_106
            ears.append((ear(kps[LEFT_EYE]) + ear(kps[RIGHT_EYE])) / 2)
            h = max(face.bbox[3] - face.bbox[1], 1)
            lips.append(np.linalg.norm(kps[UP_LIP] - kps[LO_LIP]) / h)
            aligned = norm_crop(frame, face.kps, image_size=IMAGE_SIZE)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            emotions.append(predict_emotion(gray))
        self.cap.release()

        ear_s = pd.Series(ears) if ears else pd.Series([0.5])
        lip_s = pd.Series(lips) if lips else pd.Series([0.0])
        ear_q10 = ear_s.quantile(0.10)
        lip_q90 = lip_s.quantile(0.90)
        blink_cnt = ((ear_s < ear_q10) & (ear_s.shift(1) >= ear_q10)).sum()
        blink_rate = float(blink_cnt) / (self.duration / 60.0)
        lips_apart_ratio = float((lip_s > lip_q90).mean())
        entropy = -(
            pd.Series(emotions).value_counts(normalize=True)
            .apply(lambda x: x * np.log(x + 1e-9))
        ).sum() if emotions else 0.0

        score = 0
        if blink_rate < 15: score += 1
        if entropy < 1.5: score += 1
        if lips_apart_ratio > 0.15: score += 1
        grade = min(score, 4)
        desc_map = [
            "正常：正常的面部表情",
            "轻微：仅有瞬目频率的减少",
            "轻度：除瞬目频率减少外，下面部表情也减少",
            "中度：面部表情减少，且嘴唇在不说话时偶尔张开",
            "重度：面部表情显著减少，且嘴唇在不说话时经常张开"
        ]
        return {
            "blink_rate_per_min": round(blink_rate, 2),
            "emotion_entropy": round(entropy, 3),
            "lips_apart_ratio": round(lips_apart_ratio, 3),
            "grade": int(grade),
            "description": f"{['正常', '轻微', '轻度', '中度', '重度'][grade]}：{desc_map[grade]}"
        }

def analyze_face(video_path):
    analyzer = OfflineAnalyzer(video_path)
    return analyzer.run()


def analyze_face_timeseries(video_path):
    """
    从视频中提取面部帧级时序数据，用于 LSTM 输入。
    返回:
        {
            "sequence": np.ndarray, shape=(N, 9)
                        列: [ear, lip_distance, angry, disgust, fear, happy, sad, surprise, neutral]
            "duration": float,
            "n_frames": int,
        }
    如果无法提取人脸则返回 None。
    """
    analyzer = OfflineAnalyzer(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    rows = []

    for idx in range(0, analyzer.frames, analyzer.skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = app.get(frame)
        if not faces:
            continue
        face = faces[0]
        kps = face.landmark_2d_106

        # EAR
        ear_val = (ear(kps[LEFT_EYE]) + ear(kps[RIGHT_EYE])) / 2.0

        # Lip distance (normalized by face height)
        h = max(face.bbox[3] - face.bbox[1], 1)
        lip_val = np.linalg.norm(kps[UP_LIP] - kps[LO_LIP]) / h

        # Emotion probabilities
        aligned = norm_crop(frame, face.kps, image_size=IMAGE_SIZE)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gray64 = cv2.resize(gray, (64, 64))
        blob = gray64.astype(np.float32)[None, :, :, None] / 255.0
        logits = fer_sess.run(None, {"input": blob})[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 9 维特征: [ear, lip, 7 种情绪概率]
        row = [ear_val, lip_val] + probs.tolist()
        rows.append(row)

    cap.release()

    if len(rows) == 0:
        return None

    sequence = np.array(rows, dtype=np.float32)

    return {
        "sequence": sequence,
        "duration": float(analyzer.duration),
        "n_frames": len(sequence),
    }
