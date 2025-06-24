import os
import glob
import librosa
import numpy as np
import joblib

# ===== Load trained model and scaler =====
model = joblib.load("horn_intent_model_lgbm.pkl")
scaler = joblib.load("scaler.pkl")

# ===== GCC-PHAT =====
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

# ===== Direction Estimation =====
def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# ===== Feature Extraction =====
def extract_features(left_path, right_path):
    try:
        y_left, sr_left = librosa.load(left_path, sr=None, mono=False)
        y_right, sr_right = librosa.load(right_path, sr=None, mono=False)

        if y_left.ndim == 2:
            left = y_left[0]
        else:
            left = y_left

        if y_right.ndim == 2:
            right = y_right[0]
        else:
            right = y_right

        sr = sr_left

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

        return np.hstack([left_feat, right_feat, angle]), angle
    except Exception as e:
        print(f"❗ Feature extraction failed: {e}")
        return None, None

# ===== File Paths =====
LEFT_DIR = "input/left"
RIGHT_DIR = "input/right"

left_files = glob.glob(os.path.join(LEFT_DIR, "*.wav"))
right_files = glob.glob(os.path.join(RIGHT_DIR, "*.wav"))

if not left_files or not right_files:
    print("❌ No audio files found in input/left/ or input/right/")
else:
    left_path = left_files[0]
    right_path = right_files[0]

    print(f"🎧 Left File: {left_path}")
    print(f"🎧 Right File: {right_path}")

    features, angle = extract_features(left_path, right_path)

    if features is not None:
        features_scaled = scaler.transform([features])
        predicted_intent = model.predict(features_scaled)[0]

        print(f"\n🔊 Predicted Horn Intent: {predicted_intent}")
        print(f"📐 Estimated Direction: {angle:.2f}°")

        if angle < -30:
            direction = "Right Back"
        elif -30 <= angle < -10:
            direction = "Right"
        elif -10 <= angle <= 10:
            direction = "Center Back"
        elif 10 < angle <= 30:
            direction = "Left"
        else:
            direction = "Left Back"

        print(f"📍 Final Direction Estimate: {direction}")
    else:
        print("❗ Could not process audio.")
