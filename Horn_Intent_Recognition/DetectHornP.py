import os
import joblib
import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

# === Load model and scaler ===
model = joblib.load('horn_detector_rf_model.pkl')
scaler = joblib.load('horn_rf_scaler.pkl')

# === CONFIG ===
INPUT_FOLDER = 'input'
LOWCUT = 300
HIGHCUT = 3500

# === Preprocessing ===
def bandpass_filter(signal, sr, lowcut=300, highcut=3500, order=5):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, signal)

def energy_thresholding(signal, frame_size=2048, threshold=0.01):
    energy = np.array([
        np.sum(np.abs(signal[i:i+frame_size]**2))
        for i in range(0, len(signal), frame_size)
    ])
    high_energy_frames = energy > threshold
    if np.any(high_energy_frames):
        return signal[np.repeat(high_energy_frames, frame_size)[:len(signal)]]
    return signal

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.ndim != 1:
            return None, "⚠️ Not mono audio"

        y = bandpass_filter(y, sr, LOWCUT, HIGHCUT)
        noise_clip = y[:int(sr * 0.5)]
        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)
        y = energy_thresholding(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.vstack([mfcc, delta, delta2])
        mfcc_mean = np.mean(mfcc_combined.T, axis=0)
        return mfcc_mean, None
    except Exception as e:
        return None, str(e)

# === Prediction Loop ===
print("🔍 Scanning input folder...")
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(".wav"):
        path = os.path.join(INPUT_FOLDER, filename)
        features, error = extract_features(path)
        if features is None:
            print(f"{filename}: ❌ Error - {error}")
            continue
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        print(f"{filename}: 🔊 {'Horn' if pred == 1 else 'Not Horn'}")
