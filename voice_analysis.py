import parselmouth
from parselmouth.praat import call
import numpy as np
from scipy.stats import entropy
import os

def extract_voice_features(audio_path):
    """
    Extracts 22 voice features matching the UCI Parkinson's Dataset.
    Nolds is avoided due to environment compatibility issues.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load sound
    sound = parselmouth.Sound(audio_path)
    
    # --- 1-3. F0 Features ---
    pitch = sound.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0_voiced = f0[f0 > 0]
    
    if len(f0_voiced) == 0:
        return None # Silent or unvoiced audio
        
    f0_mean = np.mean(f0_voiced)
    f0_max = np.max(f0_voiced)
    f0_min = np.min(f0_voiced)
    
    # --- 4-8. Jitter Features ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) * 100
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) * 100
    jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) * 100
    
    # --- 9-14. Shimmer Features ---
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    
    # --- 15-16. Noise Features ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 1 / (10 ** (hnr / 10)) if hnr > 0 else 0.1
    
    # --- 17-22. Non-linear & Advanced Features ---
    # DFA (Detrended Fluctuation Analysis) - Native Implementation
    dfa = calculate_dfa(f0_voiced)
    
    # RPDE (Recurrence Period Density Entropy) - Approximated
    rpde = calculate_rpde(f0_voiced)
    
    # PPE (Pitch Period Entropy)
    ppe = calculate_ppe(f0_voiced)
    
    # spread1, spread2
    spread1 = np.std(np.log(f0_voiced)) if len(f0_voiced) > 1 else 0
    spread2 = np.abs(np.mean(f0_voiced) - np.median(f0_voiced)) / np.std(f0_voiced) if np.std(f0_voiced) > 0 else 0
    
    # D2 (Correlation Dimension) - Simplified
    d2 = calculate_d2(f0_voiced)
    
    features = [
        f0_mean, f0_max, f0_min,
        jitter_local, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
        shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
        nhr, hnr,
        rpde, dfa, spread1, spread2, d2, ppe
    ]
    
    # Sanitize for JSON (Replace NaN/Inf with 0.0)
    features = [float(x) if np.isfinite(x) else 0.0 for x in features]
    
    return features

def calculate_dfa(x):
    """Calculates the Detrended Fluctuation Analysis scaling exponent."""
    if len(x) < 20: return 0.5
    N = len(x)
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(4, 10)).astype(int)
    scales = scales[scales < N/4]
    
    if len(scales) < 2: return 0.5
    
    fluctuations = []
    for s in scales:
        num_windows = N // s
        rms = []
        for i in range(num_windows):
            window = y[i*s:(i+1)*s]
            t = np.arange(s)
            poly = np.polyfit(t, window, 1)
            trend = np.polyval(poly, t)
            rms.append(np.sqrt(np.mean((window - trend)**2)))
        fluctuations.append(np.mean(rms))
        
    coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
    return coeffs[0]

def calculate_d2(x):
    """Simplified Correlation Dimension estimation."""
    if len(x) < 10: return 2.0
    # Use standard deviation of normalized signal as a proxy for complexity
    norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else x
    return 2.0 + np.std(norm)

def calculate_ppe(f0):
    if len(f0) < 2: return 0
    periods = 1.0 / f0
    semitones = 12 * np.log2(periods / (1/440.0))
    hist, _ = np.histogram(semitones, bins=20, density=True)
    return entropy(hist + 1e-12)

def calculate_rpde(f0):
    if len(f0) < 2: return 0.5
    diffs = np.diff(f0)
    return np.std(diffs) / np.mean(f0) if np.mean(f0) > 0 else 0.5

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        feat = extract_voice_features(sys.argv[1])
        print(feat)
