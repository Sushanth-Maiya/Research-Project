import os
import librosa
import numpy as np
import joblib
import json
import glob

LEFT_DIR = 'input/left'
RIGHT_DIR = 'input/right'
ODAS_OUTPUT_DIR = 'odas_output'

model = joblib.load('horn_intent_with_odas_model.pkl')
scaler = joblib.load('odas_scaler.pkl')

def get_audio_features(signal, sr):
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

def extract_features(left_file, right_file):
    try:
        y_left, sr = librosa.load(left_file, sr=None, mono=True)
        y_right, _ = librosa.load(right_file, sr=sr, mono=True)

        beam_path = os.path.join(ODAS_OUTPUT_DIR, 'beamformed.wav')
        doa_path = os.path.join(ODAS_OUTPUT_DIR, 'doa.json')

        if not os.path.exists(beam_path) or not os.path.exists(doa_path):
            print("❗ ODAS outputs missing")
            return None, None

        beam, _ = librosa.load(beam_path, sr=sr)
        with open(doa_path) as f:
            doa_data = json.load(f)
        azimuth = doa_data.get("azimuth", 0.0)

        f_left = get_audio_features(y_left, sr)
        f_right = get_audio_features(y_right, sr)
        f_beam = get_audio_features(beam, sr)

        return np.hstack([f_left, f_right, f_beam, azimuth]), azimuth
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

left_file = glob.glob(f"{LEFT_DIR}/*.wav")[0]
right_file = glob.glob(f"{RIGHT_DIR}/*.wav")[0]

print(f"🎧 Input: {left_file} + {right_file}")
features, angle = extract_features(left_file, right_file)

if features is not None:
    features_scaled = scaler.transform([features])
    intent = model.predict(features_scaled)[0]
    print(f"🔊 Predicted Intent: {intent}")
    print(f"📐 Direction: {angle:.2f}°")
else:
    print("❌ Feature extraction failed.")