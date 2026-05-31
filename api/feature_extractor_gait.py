# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import cv2, numpy as np
import math, os
from pathlib import Path

def _simple_peaks(signal):
    """
    ·µ»Ø·åË÷Òý£¨ÑÏ¸ñ¾Ö²¿¼«´óÖµ£©£¬²¢¹ýÂËÔëÉù£¨·åÖµÐè¸ßÓÚ mean + 0.2*std£©
    """
    if len(signal) < 3:
        return []
    s = np.array(signal)
    mean = np.nanmean(s)
    std = np.nanstd(s)
    thr = mean + 0.2 * std
    peaks = []
    for i in range(1, len(s)-1):
        if s[i] > s[i-1] and s[i] > s[i+1] and s[i] > thr:
            peaks.append(i)
    return peaks

def analyze_gait_from_video(video_path, sample_fps=10):
    """
    ´ÓÊÓÆµÌáÈ¡²½Ì¬ÌØÕ÷£¨Ê¹ÓÃ MediaPipe Pose if available; fallback Ê¹ÓÃ simple ¹Ø¼üµã¼ì²â²»¿ÉÓÃ -> ±¨´í£©¡£
    ·µ»ØµÄÌØÕ÷Éè¼ÆÓë gait CSV µÄ¾ÛºÏÍ³¼ÆÒ»ÖÂ£¬±ãÓÚÓÃÍ¬Ò»Ä£ÐÍÔ¤²â¡£
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise ImportError("mediapipe Î´°²×°¡£°²×°£ºpip install mediapipe opencv-python-headless") from e

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / real_fps if real_fps > 0 else total_frames / 25.0

    # sample frame step to achieve sample_fps
    step = max(1, int(round(real_fps / float(sample_fps))))
    left_ankle_x = []
    left_ankle_y = []
    right_ankle_x = []
    right_ankle_y = []
    hip_y = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks.landmark
            # mediapipe landmarks: left_hip=23, right_hip=24, left_ankle=27, right_ankle=28
            def _get(idx):
                p = lm[idx]
                return np.array([p.x * w, p.y * h, p.z if hasattr(p, 'z') else 0.0])

            la = _get(27); ra = _get(28)
            lh = _get(23); rh = _get(24)
            hip = (lh + rh) / 2.0
            left_ankle_x.append(la[0]); left_ankle_y.append(la[1])
            right_ankle_x.append(ra[0]); right_ankle_y.append(ra[1])
            hip_y.append(hip[1])

    cap.release()

    # Èç¹ûÃ»ÓÐÌáÈ¡µ½¹Ø¼üµã£¬Å×³ö¿É²¶»ñÒì³£
    if len(left_ankle_x) == 0:
        raise RuntimeError("Î´ÄÜ´ÓÊÓÆµÖÐ¼ì²âµ½×ã¹»µÄ pose ¹Ø¼üµã¡£Çë³¢ÊÔ¸üÇåÎúµÄÊÓÆµ»òÌá¸ß²ÉÑùÂÊ¡£")

    # normalize by hip width scale (mean distance between hips when available)
    hip_widths = []
    # approximate hip width from the sequence: use mean horizontal distance between left/right hips
    # We didn't store hip x separately; derive approximate scale from ankle x distance median
    ad = np.abs(np.array(left_ankle_x) - np.array(right_ankle_x))
    scale = np.median(ad) if np.median(ad) > 0 else 1.0
    # compute series: ankle distance and hip vertical displacement (relative)
    ankle_dist = ad / float(scale)
    hip_y = np.array(hip_y)
    hip_y_n = (hip_y - np.mean(hip_y)) / (np.std(hip_y) + 1e-6)

    # compute global aggregated stats (matching CSV extractor)
    def agg_stats(arr):
        arr = np.array(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0])
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'rms': float(np.sqrt(np.mean(np.square(arr)))),
            'skew': float(pd.Series(arr).skew()) if 'pd' in globals() else float(0.0),
            'kurt': float(pd.Series(arr).kurtosis()) if 'pd' in globals() else float(0.0),
            'nonzero_frac': float((arr != 0).sum() / arr.size),
            'energy': float(np.sum(np.square(arr)))
        }

    # safe import pandas used above for skew/kurtosis
    try:
        import pandas as pd
        globals()['pd'] = pd
    except Exception:
        pass

    stats_ankle = agg_stats(ankle_dist)
    stats_hip = agg_stats(hip_y_n)
    # combine data arrays to mimic gait txt 'data values'
    combined = np.concatenate([ankle_dist, hip_y_n])

    # compute peaks on ankle_dist to get steps
    peak_idx = _simple_peaks(ankle_dist)
    peak_times = [ (i * step) / (real_fps + 1e-6) for i in peak_idx ]  # seconds
    step_count = len(peak_idx)
    step_freq_per_min = float(step_count / (duration + 1e-6) * 60.0)
    peak_intervals = np.diff(peak_times) if len(peak_times) >= 2 else np.array([])

    # build feature dict that matches parse_gait_zip.py's aggregated columns
    feats = {}
    feats['duration'] = float(duration)
    feats['n_samples'] = int(len(ankle_dist))
    feats['mean_all'] = float(np.mean(combined))
    feats['std_all'] = float(np.std(combined))
    feats['median_all'] = float(np.median(combined))
    feats['min_all'] = float(np.min(combined))
    feats['max_all'] = float(np.max(combined))
    feats['mean_col_means'] = float(np.mean([stats_ankle['mean'], stats_hip['mean']]))
    feats['std_col_means']  = float(np.std([stats_ankle['mean'], stats_hip['mean']]))
    feats['mean_col_stds']  = float(np.mean([stats_ankle['std'], stats_hip['std']]))
    feats['mean_rms'] = float(np.mean([stats_ankle['rms'], stats_hip['rms']]))
    feats['skew_mean'] = float(np.mean([stats_ankle.get('skew',0.0), stats_hip.get('skew',0.0)]))
    feats['kurt_mean'] = float(np.mean([stats_ankle.get('kurt',0.0), stats_hip.get('kurt',0.0)]))
    feats['pct_nonzero'] = float((combined != 0).sum() / float(combined.size))
    feats['energy'] = float(np.sum(np.square(combined)))

    # additional gait-specific descriptors
    feats['step_count'] = int(step_count)
    feats['step_freq_per_min'] = float(step_freq_per_min)
    feats['mean_peak_interval'] = float(np.mean(peak_intervals)) if peak_intervals.size else 0.0
    feats['peak_interval_std'] = float(np.std(peak_intervals)) if peak_intervals.size else 0.0
    # symmetry: pearson correlation between left/right ankle x series (if enough samples)
    try:
        corr = float(np.corrcoef(np.array(left_ankle_x), np.array(right_ankle_x))[0,1])
        feats['ankle_corr'] = 0.0 if np.isnan(corr) else corr
    except Exception:
        feats['ankle_corr'] = 0.0

    return feats


def extract_gait_timeseries(video_path, sample_fps=10):
    """
    从视频提取步态原始帧序列（不聚合），用于 LSTM 输入。
    返回:
        {
            "sequence": np.ndarray, shape=(N, 5)
                        列: [left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, hip_y]
                        已按 body scale 归一化
            "duration": float,
            "n_frames": int,
            "step_count": int,
        }
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise ImportError("mediapipe 未安装。安装：pip install mediapipe opencv-python-headless") from e

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / real_fps if real_fps > 0 else total_frames / 25.0

    step = max(1, int(round(real_fps / float(sample_fps))))

    rows = []  # 每个 frame 一行 [la_x, la_y, ra_x, ra_y, hip_y]

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks.landmark

            la_x, la_y = lm[27].x * w, lm[27].y * h
            ra_x, ra_y = lm[28].x * w, lm[28].y * h
            lh = np.array([lm[23].x * w, lm[23].y * h])
            rh = np.array([lm[24].x * w, lm[24].y * h])
            hip_center_y = (lh[1] + rh[1]) / 2.0

            rows.append([la_x, la_y, ra_x, ra_y, hip_center_y])

    cap.release()

    if len(rows) == 0:
        raise RuntimeError("未能从视频中检测到足够的 pose 关键点。请尝试更清晰的视频或提高采样率。")

    sequence = np.array(rows, dtype=np.float32)

    # 按 body scale 归一化：用左右踝 X 距离中位数作为归一化因子
    ankle_dists = np.abs(sequence[:, 0] - sequence[:, 2])
    scale = float(np.median(ankle_dists)) if np.median(ankle_dists) > 0 else 1.0
    sequence[:, 0] /= scale   # left_ankle_x
    sequence[:, 2] /= scale   # right_ankle_x
    # Y 坐标用像素高度归一化（取自最后一帧的 h）
    # 但我们不在循环外存 h，改用列内 z-score 使其零均值
    for col in [1, 3, 4]:  # la_y, ra_y, hip_y
        col_mean = sequence[:, col].mean()
        col_std = sequence[:, col].std() + 1e-6
        sequence[:, col] = (sequence[:, col] - col_mean) / col_std

    # 步数估算
    ankle_dist_series = np.abs(sequence[:, 0] - sequence[:, 2])
    step_count = len(_simple_peaks(ankle_dist_series))

    return {
        "sequence": sequence,
        "duration": float(duration),
        "n_frames": int(len(sequence)),
        "step_count": int(step_count),
    }
