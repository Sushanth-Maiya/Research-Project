import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
AUDIO_DIR = 'audio'
EXCEL_PATH = 'horn_intents.xlsx'

# Loading Excel sheet
df = pd.read_excel(EXCEL_PATH)

# Feature Extraction Function (MFCC + Spectral Features)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Extracting MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = np.mean(spec_centroid)

        # Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_rolloff_mean = np.mean(spec_rolloff)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # Combining all features
        features = np.hstack([
            mfcc_mean,
            spec_centroid_mean,
            spec_rolloff_mean,
            zcr_mean,
            rms_mean
        ])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extracting features and labels
features = []
labels = []

print("🔍 Extracting features...")

for idx, row in df.iterrows():
    file_path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(file_path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

print(f"✅ Extracted {len(features)} samples.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Training model
print("🧠 Training model with Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("\n📊 Classification Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Accuracy as % (efficiency)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Overall Model Accuracy: {accuracy * 100:.2f}%")

# Saving the model
joblib.dump(clf, 'horn_intent_model.pkl')
print("💾 Model saved to horn_intent_model.pkl")
