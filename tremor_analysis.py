import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis

def remove_gravity(signal, fs=100):
    b, a = butter(4, 0.5/(fs/2), btype='high')
    return filtfilt(b, a, signal)

def bandpass_filter(signal, fs=100):
    b, a = butter(4, [3.0/(fs/2), 8.0/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def extract_tremor_features(x, y, z, fs=100):
    """
    Extracts 13 clinical features from 3-axis accelerometer data.
    Matches the training pipeline in 03_tremor_model.ipynb
    """
    if len(x) < 256: 
        return None
    
    # Compatibility Fix for NumPy 2.0+
    integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    
    # 1. Magnitude calculation
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    # 2. Filtering
    clean = remove_gravity(magnitude, fs)
    tremor = bandpass_filter(clean, fs)
    
    # 3. Frequency Analysis (Welch's PSD)
    freqs, power = welch(tremor, fs=fs, nperseg=min(256, len(tremor)))
    power += 1e-10 # Avoid log(0)
    total_power = integrate(power, freqs)
    
    # 4. Feature Extraction
    tremor_mask = (freqs >= 3) & (freqs <= 8)
    low_mask = (freqs < 3)
    high_mask = (freqs > 8)
    
    if np.any(tremor_mask):
        dominant_freq = freqs[tremor_mask][np.argmax(power[tremor_mask])]
        tremor_power  = integrate(power[tremor_mask], freqs[tremor_mask])
    else:
        dominant_freq, tremor_power = 0.0, 0.0
    
    tremor_ratio = tremor_power / (total_power + 1e-10)
    
    # 5. Statistical Features
    std_dev = np.std(tremor)
    max_amp = np.max(np.abs(tremor))
    mean_amp = np.mean(np.abs(tremor))
    rms = np.sqrt(np.mean(tremor**2))
    sk = float(skew(tremor))
    kt = float(kurtosis(tremor))
    
    # 6. Zero Crossing Rate
    zcr = np.sum(np.diff(np.sign(tremor)) != 0) / len(tremor)
    
    # 7. Low/High Band Power
    low_pwr = integrate(power[low_mask], freqs[low_mask]) if np.any(low_mask) else 0.0
    high_pwr = integrate(power[high_mask], freqs[high_mask]) if np.any(high_mask) else 0.0
    
    # 8. Spectral Entropy
    p_norm = power / total_power
    entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))
    
    return [
        dominant_freq, tremor_power, tremor_ratio, std_dev,
        max_amp, mean_amp, rms, sk, kt, zcr,
        low_pwr, high_pwr, entropy
    ]
