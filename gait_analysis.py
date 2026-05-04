import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
import os

def load_gait_data(filepath_or_buffer):
    """Load gait data from a file or buffer (19 columns)"""
    try:
        df = pd.read_csv(
            filepath_or_buffer,
            sep=r'\s+',
            header=None
        )
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if df.shape[1] < 19:
            return None, None, None
        
        time  = df.iloc[:, 0].values
        left  = df.iloc[:, 17].values
        right = df.iloc[:, 18].values
        
        return time, left, right
    except Exception:
        return None, None, None

def detect_steps(force_signal, fs=100):
    """Detect footstrikes from force signal"""
    if len(force_signal) < fs:
        return np.array([])
    
    b, a = butter(4, 10/(fs/2), btype='low')
    filtered = filtfilt(b, a, force_signal)
    
    mean_force = np.mean(filtered)
    std_force  = np.std(filtered)
    
    if mean_force <= 0:
        return np.array([])
    
    peaks, _ = find_peaks(
        filtered,
        height=mean_force * 0.3,
        distance=int(fs * 0.3),
        prominence=std_force * 0.3
    )
    return peaks

def extract_gait_features(filepath_or_buffer, fs=100):
    """
    Extracts 15 gait features matching the 05_gait_model notebook.
    """
    time, left, right = load_gait_data(filepath_or_buffer)
    
    if time is None or len(left) < fs * 5: # Need at least 5 seconds
        return None
    
    l_peaks = detect_steps(left,  fs)
    r_peaks = detect_steps(right, fs)
    
    if len(l_peaks) < 4 or len(r_peaks) < 4:
        return None
    
    # Stride times
    l_strides = np.diff(l_peaks) / fs
    r_strides = np.diff(r_peaks) / fs
    
    # Filter realistic strides (0.3s to 4s)
    l_strides = l_strides[(l_strides > 0.3) & (l_strides < 4.0)]
    r_strides = r_strides[(r_strides > 0.3) & (r_strides < 4.0)]
    
    if len(l_strides) < 2 or len(r_strides) < 2:
        return None
    
    all_strides = np.concatenate([l_strides, r_strides])
    
    # Feature Calculations
    mean_stride = np.mean(all_strides)
    std_stride  = np.std(all_strides)
    cv_stride   = std_stride / (mean_stride + 1e-10)
    
    cadence = ((len(l_peaks) + len(r_peaks)) / (len(left) / fs)) * 60
    
    min_len = min(len(l_strides), len(r_strides))
    symmetry = np.mean(np.abs(l_strides[:min_len] - r_strides[:min_len])) / (mean_stride + 1e-10)
    
    # Shuffle & Height
    b, a = butter(4, 10/(fs/2), btype='low')
    left_f = filtfilt(b, a, left)
    right_f = filtfilt(b, a, right)
    all_heights = np.concatenate([left_f[l_peaks], right_f[r_peaks]])
    shuffle_idx = 1 - (np.mean(all_heights) / (np.max(all_heights) + 1e-10))
    
    freeze_count = int(np.sum(all_strides > 3.0))
    freeze_ratio = freeze_count / (len(all_strides) + 1e-10)
    stride_accel = np.std(np.diff(all_strides)) if len(all_strides) > 2 else 0
    
    all_step_times = np.sort(np.concatenate([l_peaks/fs, r_peaks/fs]))
    double_support = np.mean(np.diff(all_step_times)) / (mean_stride + 1e-10)
    
    features = [
        float(mean_stride),
        float(std_stride),
        float(cv_stride),
        float(np.min(all_strides)),
        float(np.max(all_strides)),
        float(np.percentile(all_strides, 75) - np.percentile(all_strides, 25)), # iqr
        float(cadence),
        float(symmetry),
        float(shuffle_idx),
        float(freeze_count),
        float(freeze_ratio),
        float(stride_accel),
        float(double_support),
        float(skew(all_strides)),
        float(kurtosis(all_strides))
    ]
    
    return features
