import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Config
AUDIO_DIR = 'audio'
EXCEL_PATH = 'horn_intents.xlsx'

# Estimate direction angle
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')
    delay = np.argmax(corr) - (len(right) - 1)
    time_diff = delay / sr
    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None

        left, right = y[0], y[1]

        mfcc = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=left, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=left, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(left))
        rms = np.mean(librosa.feature.rms(y=left))
        angle = estimate_direction(left, right, sr)

        return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, angle])
    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None

# Load labels
df = pd.read_excel(EXCEL_PATH)
features = []
labels = []

print("🔍 Extracting features...")
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

print(f"✅ Extracted {len(features)} samples.")

# Train
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("\n📊 Classification Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save
joblib.dump(clf, 'horn_intent_model_with_angle.pkl')
print("💾 Model saved: horn_intent_model_with_angle.pkl")
