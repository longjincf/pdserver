# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import numpy as np
import parselmouth
from datetime import datetime
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def extract_features(audio_path, save_plots=False, out_dir='results'):
    # timestamped folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(out_dir, timestamp)
    if save_plots:
        os.makedirs(output_folder, exist_ok=True)

    # analysis
    sound = parselmouth.Sound(audio_path)
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    try:
        pulse_times = point_process.get_time_array()
    except AttributeError:
        n_points = parselmouth.praat.call(point_process, "Get number of points")
        pulse_times = [parselmouth.praat.call(point_process, "Get time from index", i + 1) for i in range(n_points)]

    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    plots = {}
    if save_plots:
        # waveform + pulses
        import matplotlib
        matplotlib.use('Agg')
        fig1, ax1 = plt.subplots(figsize=(12, 3))
        ax1.plot(sound.xs(), sound.values.T, alpha=0.5)
        for t in pulse_times:
            ax1.axvline(x=t, color='red', linestyle='--', linewidth=0.5)
        ax1.set_title("Waveform with Glottal Pulses")
        waveform_path = os.path.join(output_folder, "waveform_pulses.png")
        fig1.savefig(waveform_path, bbox_inches='tight', dpi=200)
        plt.close(fig1)
        plots['waveform'] = waveform_path

        # spectrogram
        spectrogram = sound.to_spectrogram()
        import numpy as np
        X, Y = np.meshgrid(spectrogram.xs(), spectrogram.ys(), indexing='ij')
        dB_levels = 10 * np.log10(spectrogram.values)
        norm = Normalize(vmin=dB_levels.min() + 20, vmax=dB_levels.max())
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.pcolormesh(X, Y, dB_levels.T, shading='auto', cmap='viridis', norm=norm)
        ax3.set_ylim(50, 2000)
        ax3.set_title("Spectrogram")
        spect_path = os.path.join(output_folder, "spectrogram.png")
        fig3.savefig(spect_path, bbox_inches='tight', dpi=200)
        plt.close(fig3)
        plots['spectrogram'] = spect_path

    return {
        "jitter_local": float(jitter),
        "shimmer_local": float(shimmer),
        "hnr_db": float(hnr),
        "plots": plots,
        "output_folder": output_folder if save_plots else None,
    }


def extract_features_timeseries(audio_path, frame_length=0.025, frame_shift=0.01):
    """
    将音频分帧，逐帧提取声学特征，用于 LSTM 输入。
    返回:
        {
            "sequence": np.ndarray, shape=(N, 3)
                        列: [jitter_local, shimmer_local, hnr_db] 逐帧
            "duration": float,
            "n_frames": int,
        }

    注意: Parselmouth 的 Jitter/Shimmer 需要 PointProcess，
    逐帧提取开销较大，这里对每帧（若干周期的信号）计算局部值。
    对于过短的帧（无法产生 PointProcess），用前一个有效值填充。
    """
    sound = parselmouth.Sound(audio_path)
    total_duration = sound.get_total_duration()
    n_frames = int(total_duration / frame_shift) + 1

    jitter_list = []
    shimmer_list = []
    hnr_list = []

    prev_jitter = 0.0
    prev_shimmer = 0.0
    prev_hnr = 0.0

    for i in range(n_frames):
        t_start = i * frame_shift
        t_end = t_start + frame_length
        if t_end > total_duration:
            t_end = total_duration

        try:
            # 截取当前帧
            frame_sound = sound.extract_part(from_time=t_start, to_time=t_end,
                                             preserve_times=True)
            # Jitter / Shimmer 需要 PointProcess
            pp = parselmouth.praat.call(frame_sound, "To PointProcess (periodic, cc)", 75, 600)
            j = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            s = parselmouth.praat.call([frame_sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = parselmouth.praat.call(frame_sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            h = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

            # 过滤异常值
            j = max(0.0, min(j, 5.0)) if not np.isnan(j) else prev_jitter
            s = max(0.0, min(s, 5.0)) if not np.isnan(s) else prev_shimmer
            h = h if not np.isnan(h) else prev_hnr

            prev_jitter, prev_shimmer, prev_hnr = j, s, h
            jitter_list.append(j)
            shimmer_list.append(s)
            hnr_list.append(h)
        except Exception:
            # 当前帧太短或无基音，使用上一个有效值
            jitter_list.append(prev_jitter)
            shimmer_list.append(prev_shimmer)
            hnr_list.append(prev_hnr)

    sequence = np.column_stack([
        np.array(jitter_list, dtype=np.float32),
        np.array(shimmer_list, dtype=np.float32),
        np.array(hnr_list, dtype=np.float32),
    ])

    return {
        "sequence": sequence,
        "duration": float(total_duration),
        "n_frames": int(n_frames),
    }
