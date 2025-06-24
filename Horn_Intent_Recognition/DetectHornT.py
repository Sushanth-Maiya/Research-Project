import os
import librosa
import numpy as np
import pandas as pd
import joblib
import noisereduce as nr
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === CONFIG ===
EXCEL_PATH = 'horn_intents_updated.xlsx'
AUDIO_DIR = 'audio'
LABEL_COLUMN = 'horn'  # Column name in Excel for horn labels (1 or 0)
LOWCUT = 300           # Bandpass filter low cutoff (Hz)
HIGHCUT = 3500         # Bandpass filter high cutoff (Hz)

# === Preprocessing ===
def bandpass_filter(signal, sr, lowcut=300, highcut=3500, order=5):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
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
        y = bandpass_filter(y, sr, LOWCUT, HIGHCUT)
        noise_clip = y[:int(sr * 0.5)]
        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)
        y = energy_thresholding(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.vstack([mfcc, delta, delta2])
        mfcc_mean = np.mean(mfcc_combined.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# === Load dataset ===
df = pd.read_excel(EXCEL_PATH)
features, labels = [], []

for _, row in df.iterrows():
    filename = row['filename']
    label = row[LABEL_COLUMN]
    path = os.path.join(AUDIO_DIR, filename)
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(label)

X = np.array(features)
y = np.array(labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train Random Forest ===
print("🌲 Training Random Forest horn detector...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# === Save model ===
joblib.dump(model, 'horn_detector_rf_model.pkl')
joblib.dump(scaler, 'horn_rf_scaler.pkl')
print("💾 Model saved as horn_detector_rf_model.pkl")
