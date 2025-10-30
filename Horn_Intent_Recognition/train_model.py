# ====== TRAINING SCRIPT: horn intent classification + direction estimation ======

import os
import pandas as pd
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import butter, lfilter

AUDIO_DIR = 'audio'
EXCEL_PATH = 'horn_intents.xlsx'

# ===== Bandpass Filter =====
def bandpass_filter(signal, sr, lowcut=250, highcut=5000, order=6):
    nyquist = 0.5 * sr
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return lfilter(b, a, signal)

# ===== Amplitude in dBFS =====
def compute_amplitude_db(signal):
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms + 1e-6)  # avoid log(0)

# ===== GCC-PHAT with sub-sample precision =====
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

def extract_features(file_path, amplitude_threshold_db=-45):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim < 2:
            return None

        # Amplitude check (reject weak signals)
        db_left = compute_amplitude_db(y[0])
        db_right = compute_amplitude_db(y[1])
        if db_left < amplitude_threshold_db and db_right < amplitude_threshold_db:
            return None

        left = bandpass_filter(y[0], sr)
        right = bandpass_filter(y[1], sr)

        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
            rms = np.mean(librosa.feature.rms(y=signal))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])

        left_feat = get_audio_features(left)
        right_feat = get_audio_features(right)
        angle = estimate_direction(left, right, sr)
        return np.hstack([left_feat, right_feat, angle])
    except:
        return None

# ===== Load Data and Train =====
df = pd.read_excel(EXCEL_PATH)
features, labels = [], []

for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(best_model, 'horn_intent_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Model saved.")
