import os
import glob
import joblib
import librosa
import numpy as np
from scipy.signal import butter, lfilter

# ===== Bandpass Filter (Change here for dB filter range) =====
def bandpass_filter(signal, sr, lowcut=300, highcut=3000, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, signal)

# ===== Sub-sample GCC-PHAT =====
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

def extract_features(left_path, right_path):
    try:
        y_left, sr = librosa.load(left_path, sr=None)
        y_right, _ = librosa.load(right_path, sr=None)

        left = bandpass_filter(y_left, sr)
        right = bandpass_filter(y_right, sr)

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
    except:
        return None, None

# ===== Prediction =====
print("🔍 Loading model...")
model = joblib.load("horn_intent_model.pkl")
scaler = joblib.load("scaler.pkl")

LEFT_AUDIO = glob.glob('input/left/*.wav')
RIGHT_AUDIO = glob.glob('input/right/*.wav')

if not LEFT_AUDIO or not RIGHT_AUDIO:
    print("❌ Missing input files.")
else:
    left_file = LEFT_AUDIO[0]
    right_file = RIGHT_AUDIO[0]
    print(f"🎧 Files:\n  Left: {left_file}\n  Right: {right_file}")

    features, angle = extract_features(left_file, right_file)
    if features is not None:
        scaled = scaler.transform([features])
        intent = model.predict(scaled)[0]
        print(f"\n🔊 Predicted Horn Intent: {intent}")
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

        print(f"📍 Estimated Horn Direction: {direction}")
    else:
        print("❗ Feature extraction failed.")
