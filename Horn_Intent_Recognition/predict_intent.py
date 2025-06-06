import os
import glob
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ==== Paths ====
MODEL_PATH = 'horn_intent_model.pkl'
SCALER_PATH = 'scaler.pkl'
LEFT_AUDIO = glob.glob('input/left/*.wav')
RIGHT_AUDIO = glob.glob('input/right/*.wav')

# ==== Load model ====
print("🔍 Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==== Direction Estimation ====
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')
    delay = np.argmax(corr) - (len(right) - 1)
    time_diff = delay / sr
    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# ==== Feature Extraction ====
def extract_features(left_path, right_path):
    try:
        y_left, sr_left = librosa.load(left_path, sr=None, mono=False)
        y_right, sr_right = librosa.load(right_path, sr=None, mono=False)

        # Convert to mono if stereo
        if y_left.ndim == 2:
            left = y_left[0]
        else:
            left = y_left

        if y_right.ndim == 2:
            right = y_right[0]
        else:
            right = y_right

        sr = sr_left  # Assume both have same sample rate

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

        features = np.hstack([left_feat, right_feat, angle])
        return features, angle

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None, None

# ==== Predict ====
if not LEFT_AUDIO or not RIGHT_AUDIO:
    print("❌ No audio files found in input/left/ or input/right/")
else:
    left_file = LEFT_AUDIO[0]
    right_file = RIGHT_AUDIO[0]
    print(f"🎧 Left: {left_file}")
    print(f"🎧 Right: {right_file}")

    features, angle = extract_features(left_file, right_file)

    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        print(f"\n🔊 Predicted Intent: **{prediction}**")
        print(f"📐 Estimated Direction: {angle:.2f}°")

        # ==== Polar Plot ====
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.plot([0, np.radians(angle)], [0, 1], marker='o', color='red', linewidth=2)
        ax.set_title(f"Direction: {angle:.2f}°", va='bottom')
        plt.show()
    else:
        print("⚠️ Could not extract features.")
