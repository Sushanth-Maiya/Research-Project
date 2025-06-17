# ===== Import necessary libraries =====
import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===== Define paths =====
AUDIO_DIR = 'audio'
EXCEL_PATH = 'horn_intents.xlsx'

# ===== Function: GCC-PHAT for Time Delay Estimation =====
# Function Name: gcc_phat
# Purpose: Calculates the time delay between two signals using the GCC-PHAT method
# Inputs:
#   - sig: First audio signal (numpy array)
#   - refsig: Second audio signal (numpy array)
#   - fs: Sampling rate (default is 1)
#   - max_tau: Optional maximum time delay limit
#   - interp: Interpolation factor (default is 16)
# Output:
#   - Estimated time delay (tau) between the two signals

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

# ===== Function: estimate_direction =====
# Function Name: estimate_direction
# Purpose: Estimates the direction angle based on the time delay between two microphones
# Inputs:
#   - left: Audio signal from the left microphone (numpy array)
#   - right: Audio signal from the right microphone (numpy array)
#   - sr: Sampling rate of the audio
#   - mic_distance: Distance between microphones (default is 0.2 meters)
# Output:
#   - Estimated direction angle in degrees

def estimate_direction(left, right, sr, mic_distance=0.2):
    tau = gcc_phat(left, right, fs=sr)
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)
    except:
        return 0.0

# ===== Function: extract_features =====
# Function Name: extract_features
# Purpose: Extracts audio features from a stereo audio file and estimates the direction angle
# Inputs:
#   - file_path: Path to the stereo audio file
# Output:
#   - Combined feature vector with left and right channel features and estimated angle

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)

        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None

        left, right = y[0], y[1]

        # ===== Internal Function: get_audio_features =====
        # Function Name: get_audio_features
        # Purpose: Extracts MFCCs, spectral, and tonal features from an audio signal
        # Input:
        #   - signal: Audio signal array
        # Output:
        #   - Combined feature vector

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

        return np.hstack([left_feat, right_feat, angle])
    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None

# ===== Read the Excel file containing filenames and labels =====
df = pd.read_excel(EXCEL_PATH)

features = []
labels = []

print("🔍 Extracting features...")
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(row['intent'])

print(f"✅ Extracted {len(features)} samples.")

# ===== Prepare the dataset =====
X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===== Train the model using GridSearch =====
print("🔧 Training model with GridSearch...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ===== Evaluate the model =====
print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ===== Save the model and scaler =====
joblib.dump(best_model, 'horn_intent_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Model saved: horn_intent_model.pkl")