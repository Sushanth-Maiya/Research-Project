import os
import joblib
import librosa
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = 'horn_intent_model.pkl'
AUDIO_FILE = 'Horn3.wav'   # <- Replace with your test file name
# ----------------------------------------

# Load model
print("🔍 Loading trained model...")
model = joblib.load(MODEL_PATH)

# Extract features from the new file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # MFCC (13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Spectral Centroid (1)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)

        # Spectral Rolloff (1)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)

        # Zero Crossing Rate (1)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # RMS Energy (1)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # Combine all features (13 + 1 + 1 + 1 + 1 = 17)
        combined = np.hstack([mfcc_mean, centroid_mean, rolloff_mean, zcr_mean, rms_mean])
        return combined

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Check if file exists
if not os.path.exists(AUDIO_FILE):
    print(f"❌ File '{AUDIO_FILE}' not found.")
else:
    print(f"Predicting for file: {AUDIO_FILE}")
    feat = extract_features(AUDIO_FILE)

    if feat is not None:
        feat = feat.reshape(1, -1)  # reshape for prediction
        prediction = model.predict(feat)[0]
        print(f"\nPredicted Intent: **{prediction}**")
    else:
        print("⚠️ Could not extract features.")
