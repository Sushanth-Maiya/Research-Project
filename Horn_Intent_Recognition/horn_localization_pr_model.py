import os
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Config
MODEL_PATH = 'horn_intent_model_with_angle.pkl'
AUDIO_FILE = 'Horn4.wav'  # ← must be stereo

# Load model
print("🔍 Loading model...")
model = joblib.load(MODEL_PATH)

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

# Extract features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim < 2:
            print("❗ Stereo file required.")
            return None, None

        left, right = y[0], y[1]
        mfcc = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=left, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=left, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(left))
        rms = np.mean(librosa.feature.rms(y=left))
        angle = estimate_direction(left, right, sr)

        combined = np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, angle])
        return combined, angle
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# Predict
if not os.path.exists(AUDIO_FILE):
    print(f"❌ File not found: {AUDIO_FILE}")
else:
    print(f"🎧 Analyzing {AUDIO_FILE}...")
    features, angle = extract_features(AUDIO_FILE)

    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        print(f"\n🔊 Predicted Intent: **{prediction}**")
        print(f"📐 Estimated Direction: {angle:.2f}°")

        # Plot direction
        plt.figure()
        plt.polar([0, np.radians(angle)], [0, 1], marker='o')
        plt.title(f"Direction: {angle:.2f}°")
        plt.show()
    else:
        print("⚠️ Feature extraction failed.")
