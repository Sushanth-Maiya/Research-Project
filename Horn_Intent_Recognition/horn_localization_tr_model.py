# ===== Import necessary libraries =====
import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===== Define paths =====
AUDIO_DIR = 'audio'  # Folder containing the audio files
EXCEL_PATH = 'horn_intents_updated.xlsx'  # Updated Excel file

# ===== Function: GCC-PHAT for Time Delay Estimation =====
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

# ===== Function: estimate_direction =====
def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            return 'Back Left'
        else:
            return 'Back Right'
    except:
        return 'Unknown'

# ===== Function: extract audio features =====
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)

        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None, None

        left, right = y[0], y[1]

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

        direction = estimate_direction(left, right, sr)

        combined_features = np.hstack([left_feat, right_feat])
        return combined_features, direction

    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None, None

# ===== Load Excel file =====
df = pd.read_excel(EXCEL_PATH)

# ===== Prepare data =====
features = []
horn_labels = []
intent_labels = []
direction_labels = []

print("🔍 Extracting features...")

for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])
    feat, direction = extract_features(path)

    if feat is not None:
        features.append(feat)
        horn_labels.append('Horn' if row['horn_presence'] == 1 else 'Not Horn')  # "Horn" or "Not Horn"

        if row['horn_presence'] == 1:
            intent_labels.append(row['intent'])  # Intent label for horn
            direction_labels.append(direction)
        else:
            intent_labels.append('None')
            direction_labels.append('None')

print(f"✅ Extracted {len(features)} samples.")

# ===== Convert to numpy arrays =====
X = np.array(features)
y_horn = np.array(horn_labels)
y_intent = np.array(intent_labels)
y_direction = np.array(direction_labels)

# ===== Scale features =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== Train Horn / Not Horn Classifier =====
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_horn, test_size=0.2, random_state=42)

print("🔧 Training Horn/Not Horn classifier...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

grid_horn = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_horn.fit(X_train, y_train)
best_horn_model = grid_horn.best_estimator_

print("\n📊 Horn/Not Horn Classification Report:")
y_pred = best_horn_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(best_horn_model, 'horn_detection_model.pkl')
print("💾 Horn/Not Horn model saved: horn_detection_model.pkl")

# ===== Prepare Horn-only samples for Intent and Direction classification =====
horn_indices = np.where(y_horn == 'Horn')[0]

X_horn = X_scaled[horn_indices]
y_intent_horn = y_intent[horn_indices]
y_direction_horn = y_direction[horn_indices]

# ===== Train Intent Classifier =====
X_train_intent, X_test_intent, y_train_intent, y_test_intent = train_test_split(X_horn, y_intent_horn, test_size=0.2, random_state=42)

print("🔧 Training Horn Intent classifier...")
grid_intent = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_intent.fit(X_train_intent, y_train_intent)
best_intent_model = grid_intent.best_estimator_

print("\n📊 Horn Intent Classification Report:")
y_pred_intent = best_intent_model.predict(X_test_intent)
print(classification_report(y_test_intent, y_pred_intent))
print(f"✅ Accuracy: {accuracy_score(y_test_intent, y_pred_intent) * 100:.2f}%")

joblib.dump(best_intent_model, 'horn_intent_model.pkl')
print("💾 Horn Intent model saved: horn_intent_model.pkl")

# ===== Train Direction Classifier =====
X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(X_horn, y_direction_horn, test_size=0.2, random_state=42)

print("🔧 Training Horn Direction classifier...")
grid_dir = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_dir.fit(X_train_dir, y_train_dir)
best_dir_model = grid_dir.best_estimator_

print("\n📊 Horn Direction Classification Report:")
y_pred_dir = best_dir_model.predict(X_test_dir)
print(classification_report(y_test_dir, y_pred_dir))
print(f"✅ Accuracy: {accuracy_score(y_test_dir, y_pred_dir) * 100:.2f}%")

joblib.dump(best_dir_model, 'horn_direction_model.pkl')
print("💾 Horn Direction model saved: horn_direction_model.pkl")

# ===== Save Scaler =====
joblib.dump(scaler, 'feature_scaler.pkl')
print("💾 Scaler saved: feature_scaler.pkl")
