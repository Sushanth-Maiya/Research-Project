# Import required libraries
import os  # To work with file paths and directories
import pandas as pd  # To read and work with Excel files
import librosa  # Audio analysis and feature extraction
import numpy as np  # Numerical computing (arrays, math functions)
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
from sklearn.ensemble import RandomForestClassifier  # Our ML model
from sklearn.metrics import classification_report, accuracy_score  # For evaluating the model
import joblib  # For saving and loading trained models

# === Step 1: Set paths to data ===
AUDIO_DIR = 'audio'  # Folder where audio files are stored
EXCEL_PATH = 'horn_intents.xlsx'  # Excel file mapping audio filenames to horn intents

# === Step 2: Load labels from Excel file ===
# The Excel file must have at least two columns: 'filename' and 'intent'
df = pd.read_excel(EXCEL_PATH)

# === Step 3: Function to extract audio and localization features ===
def extract_features(file_path):
    try:
        # Load stereo audio. mono=False keeps left and right channels separate
        y, sr = librosa.load(file_path, sr=None, mono=False)
        left = y[0]  # Left channel audio data
        right = y[1]  # Right channel audio data

        # ========== Audio Features (from LEFT channel only) ==========

        # 1. MFCCs (Mel-Frequency Cepstral Coefficients): represent timbre
        mfcc = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Take mean across time

        # 2. Spectral Centroid: where the "center of mass" of sound is
        centroid = librosa.feature.spectral_centroid(y=left, sr=sr)
        centroid_mean = np.mean(centroid)

        # 3. Spectral Rolloff: frequency below which most energy is concentrated
        rolloff = librosa.feature.spectral_rolloff(y=left, sr=sr)
        rolloff_mean = np.mean(rolloff)

        # 4. Zero Crossing Rate: number of times waveform crosses zero
        zcr = librosa.feature.zero_crossing_rate(y=left)
        zcr_mean = np.mean(zcr)

        # 5. RMS Energy: average energy (loudness)
        rms = librosa.feature.rms(y=left)
        rms_mean = np.mean(rms)

        # ========== Sound Localization Feature ==========

        # Estimate Time Delay between left and right signals using cross-correlation
        corr = np.correlate(left, right, mode='full')  # Cross-correlation
        lag = np.argmax(corr) - len(right) + 1  # Time lag in samples
        time_diff = lag / sr  # Convert lag to time (seconds)

        # Use time difference to estimate direction (left: -1, right: +1)
        direction = np.clip(time_diff * 343.0, -1.0, 1.0)  # 343 m/s = speed of sound

        # Combine all features into one array
        features = np.hstack([
            mfcc_mean,         # 13 MFCCs
            centroid_mean,     # 1 value
            rolloff_mean,      # 1 value
            zcr_mean,          # 1 value
            rms_mean,          # 1 value
            direction          # 1 value from localization
        ])
        return features

    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {e}")
        return None

# === Step 4: Extract features and labels for all audio files ===
features = []
labels = []

print("🔍 Extracting features from all audio files...")

for idx, row in df.iterrows():
    file_path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(file_path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

print(f"✅ Features extracted for {len(features)} files.")

# === Step 5: Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# === Step 6: Train a Random Forest classifier ===
print("🧠 Training the Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Step 7: Evaluate the model ===
print("\n📊 Classification Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Print overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# === Step 8: Save the trained model ===
joblib.dump(clf, 'horn_intent_model.pkl')
print("💾 Model saved as horn_intent_model.pkl")
