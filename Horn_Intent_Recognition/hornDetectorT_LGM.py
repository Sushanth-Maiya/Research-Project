import os
import librosa
import numpy as np
import pandas as pd
import joblib
import noisereduce as nr
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# Config
EXCEL_PATH = 'horn_intents_updated.xlsx'
AUDIO_DIR = 'audio'
LABEL_COLUMN = 'horn_presence'
LOWCUT, HIGHCUT = 300, 3500

def bandpass_filter(signal, sr):
    nyq = 0.5 * sr
    b, a = butter(5, [LOWCUT/nyq, HIGHCUT/nyq], btype='band')
    return lfilter(b, a, signal)

def energy_thresholding(signal):
    frame_size, threshold = 2048, 0.01
    energy = np.array([np.sum(np.abs(signal[i:i+frame_size]**2))
                       for i in range(0, len(signal), frame_size)])
    mask = energy > threshold
    return signal[np.repeat(mask, frame_size)[:len(signal)]] if np.any(mask) else signal

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=None)
        y = bandpass_filter(y, sr)
        y = nr.reduce_noise(y=y, sr=sr, y_noise=y[:sr//2])
        y = energy_thresholding(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < 9:
            mfcc = np.pad(mfcc, ((0,0),(0,9-mfcc.shape[1])), mode='edge')
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        return np.mean(combined.T, axis=0)
    except Exception as e:
        print(f"Error: {path} -> {e}")
        return None

df = pd.read_excel(EXCEL_PATH)
X, y = [], []
for _, r in df.iterrows():
    feat = extract_features(os.path.join(AUDIO_DIR, r['filename']))
    if feat is not None:
        X.append(feat); y.append(r[LABEL_COLUMN])

X = np.array(X); y = np.array(y)
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
col_names = [f"f{i}" for i in range(Xs.shape[1])]
Xdf = pd.DataFrame(Xs, columns=col_names)

Xtr, Xte, ytr, yte = train_test_split(Xdf, y, test_size=0.2, random_state=42)
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                           max_depth=7, num_leaves=31,
                           random_state=42, verbosity=-1)
model.fit(Xtr, ytr)

yp = model.predict(Xte)
print(classification_report(yte, yp))
print("Acc:", accuracy_score(yte, yp))

joblib.dump(model, 'horn_lgbm.pkl')
joblib.dump(scaler, 'horn_scaler.pkl')
print("✔ LightGBM model saved.")
