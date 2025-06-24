import os
import glob
import librosa
import numpy as np
import joblib

# Load model and scaler
print("🔍 Loading model and scaler...")
model = joblib.load('horn_detection_model.pkl')  # Load Horn/Not Horn model
scaler = joblib.load('scaler.pkl')              # Load the scaler

# ===== Function: GCC-PHAT for Time Delay Estimation =====
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

# ===== Function: Estimate Direction =====
def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# ===== Function: Extract Features =====
def extract_features(left_path, right_path):
    try:
        y_left, sr_left = librosa.load(left_path, sr=None, mono=False)
        y_right, sr_right = librosa.load(right_path, sr=None, mono=False)

        print(f"🎧 Left audio shape: {y_left.shape}")
        print(f"🎧 Right audio shape: {y_right.shape}")
        print(f"🎼 Sampling Rates: Left = {sr_left} Hz, Right = {sr_right} Hz")

        if y_left.ndim == 2:
            left = y_left[0]
        else:
            left = y_left

        if y_right.ndim == 2:
            right = y_right[0]
        else:
            right = y_right

        sr = sr_left  # Assuming both have the same sampling rate

        print(f"📈 Left signal mean: {np.mean(left)}, std: {np.std(left)}")
        print(f"📈 Right signal mean: {np.mean(right)}, std: {np.std(right)}")

        # Internal function to extract features
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

        print(f"📐 Estimated Angle: {angle:.2f}°")

        combined_features = np.hstack([left_feat, right_feat])

        print(f"🔍 Extracted Feature Vector: {combined_features}")
        print(f"🔍 Feature Vector Shape: {combined_features.shape}")

        return combined_features, angle

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None, None

# ===== Get Input Files =====
LEFT_AUDIO = glob.glob('input/left/*.wav')
RIGHT_AUDIO = glob.glob('input/right/*.wav')

if not LEFT_AUDIO or not RIGHT_AUDIO:
    print("❌ No audio files found in input/left/ or input/right/ directory.")
else:
    left_file = LEFT_AUDIO[0]
    right_file = RIGHT_AUDIO[0]

    print(f"🎧 Left Input File: {left_file}")
    print(f"🎧 Right Input File: {right_file}")

    features, angle = extract_features(left_file, right_file)

    if features is not None:
        features_scaled = scaler.transform([features])
        print(f"🧮 Scaled Feature Vector: {features_scaled}")

        predicted_horn = model.predict(features_scaled)[0]
        print(f"\n🔊 Horn Detection Result: {predicted_horn}")

        if predicted_horn == 'Horn':
            print("✅ Horn detected in input audio.")
        else:
            print("🚫 No horn detected. Please verify your input audio or check feature compatibility.")

    else:
        print("❗ Feature extraction failed.")