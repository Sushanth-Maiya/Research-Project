# ===== Import necessary libraries =====
import os  # For interacting with the operating system and file paths
import pandas as pd  # For reading and manipulating the Excel file containing labels
import librosa  # For audio loading and feature extraction
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # For the classification model
from sklearn.preprocessing import StandardScaler  # For normalizing the feature vectors
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance
import joblib  # For saving and loading trained models
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, sr, lowcut=300, highcut=3500, order=4):
    """Applies a bandpass filter to isolate frequencies between lowcut and highcut Hz.
    You can change lowcut/highcut values here if needed.
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ===== Define paths =====
AUDIO_DIR = 'audio'  # Folder containing the audio files
EXCEL_PATH = 'horn_intents.xlsx'  # Excel file containing filenames and labels

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
    # """
    # Computes the time delay between sig and refsig using GCC-PHAT with:
    # - Sub-sample precision using parabolic interpolation
    # - Bandpass filter to suppress irrelevant noise

    # Returns: Estimated delay (tau)
    # """

    # Bandpass filter both signals
    sig = bandpass_filter(sig, fs)
    refsig = bandpass_filter(refsig, fs)

    # Pad signals and compute FFT
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift

    # --- Sub-sample precision using parabolic interpolation ---
    if 1 <= shift + max_shift < len(cc) - 1:
        y1 = cc[shift + max_shift - 1]
        y2 = cc[shift + max_shift]
        y3 = cc[shift + max_shift + 1]
        denom = y1 - 2 * y2 + y3
        if denom != 0:
            shift = shift + 0.5 * (y1 - y3) / denom

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
    # Estimate angle of arrival based on time delay and microphone distance
    tau = gcc_phat(left, right, fs=sr)  # Get time delay between channels
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))  # Calculate angle in radians
        return np.degrees(angle_rad)  # Convert to degrees
    except:
        return 0.0  # Fallback if calculation fails

# ===== Function: extract_features =====
# Function Name: extract_features
# Purpose: Extracts audio features from a stereo audio file and estimates the direction angle
# Inputs:
#   - file_path: Path to the stereo audio file
# Output:
#   - Combined feature vector with left and right channel features and estimated angle

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)  # Load stereo audio without resampling

        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")  # Error message for mono audio
            return None

        left, right = y[0], y[1]  # Split stereo channels

                # ===== Internal Function: get_audio_features =====
        # Function Name: get_audio_features
        # Purpose: Extracts MFCCs, spectral, and tonal features from an audio signal
        # Input:
        #   - signal: Audio signal array
        # Output:
        #   - Combined feature vector

        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # MFCCs
            mfcc_mean = np.mean(mfcc.T, axis=0)  # Mean MFCCs over time
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))  # Spectral centroid
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))  # Spectral rolloff
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))  # Zero-crossing rate
            rms = np.mean(librosa.feature.rms(y=signal))  # Root Mean Square energy
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))  # Spectral bandwidth
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))  # Chroma features
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))  # Tonnetz (tonal centroid)
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])  # Concatenate all features

        left_feat = get_audio_features(left)  # Features from left channel
        right_feat = get_audio_features(right)  # Features from right channel

        angle = estimate_direction(left, right, sr)  # Directional angle from stereo audio

        return np.hstack([left_feat, right_feat, angle])  # Combine all features
    except Exception as e:
        print(f"Error: {file_path} -> {e}")  # Handle unexpected errors
        return None

# ===== Load Excel file with labels =====
df = pd.read_excel(EXCEL_PATH)  # Read Excel containing filename and intent label

features = []  # List to store feature vectors
labels = []  # List to store corresponding labels

print("🔍 Extracting features...")
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])  # Construct full file path
    feat = extract_features(path)  # Extract features from the file
    if feat is not None:
        features.append(feat)  # Add features to list
        labels.append(row['intent'])  # Add label to list

print(f"✅ Extracted {len(features)} samples.")  # Print how many samples were processed

# ===== Prepare dataset for training =====
X = np.array(features)  # Convert feature list to numpy array
y = np.array(labels)  # Convert label list to numpy array

scaler = StandardScaler()  # Create a scaler object for normalization
X_scaled = scaler.fit_transform(X)  # Normalize features to zero mean and unit variance

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Split data

# ===== Train model using Random Forest + GridSearch =====
print("🔧 Training model with GridSearch...")
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [None, 20, 40],  # Tree depth
    'min_samples_split': [2, 5],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2],  # Minimum samples per leaf
    'class_weight': [None, 'balanced']  # Handle imbalanced classes
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)  # Grid search setup
grid.fit(X_train, y_train)  # Train with cross-validation

best_model = grid.best_estimator_  # Get the best model from GridSearch

# ===== Evaluate model =====
print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)  # Predict on test set
print(classification_report(y_test, y_pred))  # Show precision, recall, f1-score
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")  # Show accuracy

# ===== Save the model and scaler =====
joblib.dump(best_model, 'horn_intent_model.pkl')  # Save the model to disk
joblib.dump(scaler, 'scaler.pkl')  # Save the feature scaler
print("💾 Model saved: horn_intent_model.pkl")  # Confirmation message
