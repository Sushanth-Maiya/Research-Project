import os
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = 'horn_intent_with_sound_localization_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
AUDIO_FILE = 'Horn4.wav'

# Load model and scaler
print("🔍 Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Estimate direction from stereo audio
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')
    delay = np.argmax(corr) - (len(right) - 1)
    time_diff = delay / sr
    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# Feature extraction (full version using both channels)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim < 2:
            print("❗ Stereo file required.")
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
        angle = estimate_direction(left, right, sr)
        full_features = np.hstack([left_feat, right_feat, angle])

        return full_features, angle
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# Prediction process
if not os.path.exists(AUDIO_FILE):
    print(f"❌ File not found: {AUDIO_FILE}")
else:
    print(f"🎧 Analyzing {AUDIO_FILE}...")
    features, angle = extract_features(AUDIO_FILE)

    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        print(f"\n🔊 Predicted Intent: **{prediction}**")
        print(f"📐 Estimated Direction: {angle:.2f}°")

        # Plotting direction
        plt.figure()
        plt.polar([0, np.radians(angle)], [0, 1], marker='o', color='red')
        plt.title(f"Direction: {angle:.2f}°")
        plt.show()
    else:
        print("⚠️ Feature extraction failed.")
