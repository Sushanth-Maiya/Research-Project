# train_with_odas.py

import os
import numpy as np
import pandas as pd
import librosa
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

EXCEL_PATH = 'horn_intents.xlsx'
AUDIO_DIR = 'audio'
ODAS_OUTPUT_DIR = 'odas_output'

def get_audio_features(signal, sr):
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

def extract_features(filename):
    try:
        path = os.path.join(AUDIO_DIR, filename)
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim < 2:
            print(f"❗ Not stereo: {filename}")
            return None
        left, right = y[0], y[1]

        beam_path = os.path.join(ODAS_OUTPUT_DIR, filename.replace('.wav', '_beam.wav'))
        doa_path = os.path.join(ODAS_OUTPUT_DIR, filename.replace('.wav', '_doa.json'))

        if not os.path.exists(beam_path) or not os.path.exists(doa_path):
            print(f"❗ ODAS outputs missing for {filename}")
            return None

        beam, _ = librosa.load(beam_path, sr=sr)
        with open(doa_path) as f:
            doa_data = json.load(f)
        azimuth = doa_data.get("azimuth", 0.0)

        feat_left = get_audio_features(left, sr)
        feat_right = get_audio_features(right, sr)
        feat_beam = get_audio_features(beam, sr)

        return np.hstack([feat_left, feat_right, feat_beam, azimuth])
    except Exception as e:
        print(f"❌ Error in {filename}: {e}")
        return None

# Load Excel
df = pd.read_excel(EXCEL_PATH)
features = []
labels = []

print("🔍 Extracting features with ODAS...")
for _, row in df.iterrows():
    feat = extract_features(row['filename'])
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("🔧 Training model...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_
y_pred = model.predict(X_test)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(model, 'horn_intent_with_odas_model.pkl')
joblib.dump(scaler, 'odas_scaler.pkl')
print("💾 Model saved.")
