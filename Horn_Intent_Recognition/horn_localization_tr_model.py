import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

AUDIO_DIR = 'audio'
EXCEL_PATH = 'horn_intents.xlsx'

def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')
    delay = np.argmax(corr) - (len(right) - 1)
    time_diff = delay / sr
    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None

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
        angle = estimate_direction(left, right, sr)

        return np.hstack([left_feat, right_feat, angle])
    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None

df = pd.read_excel(EXCEL_PATH)
features, labels = [], []

print("🔍 Extracting features...")
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

print(f"✅ Extracted {len(features)} samples.")

X = np.array(features)
y = np.array(labels)

# 🔄 Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🤖 Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔍 Optional: Grid search for best parameters
print("🔧 Training model with GridSearch...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

joblib.dump(best_model, 'horn_intent_with_sound_localization_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("💾 Model saved: horn_intent_with_sound_localization_model.pkl")
