# === Import required libraries ===

import os  # For interacting with the file system
import pandas as pd  # For reading Excel files and handling tabular data
import librosa  # For loading audio files and extracting features
import numpy as np  # For numerical operations and array handling
from sklearn.model_selection import train_test_split, GridSearchCV  # For dataset splitting and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # Machine learning model: Random Forest
from sklearn.preprocessing import StandardScaler  # For feature normalization (scaling)
from sklearn.metrics import classification_report, accuracy_score  # For model evaluation
import joblib  # For saving and loading the trained model and scaler

# === Constants ===

AUDIO_DIR = 'audio'  # Folder containing stereo .wav audio files
EXCEL_PATH = 'horn_intents.xlsx'  # Excel file with 'filename' and 'intent' columns

# === Function: estimate_direction ===
# Purpose: Estimate the angle of sound arrival (azimuth) based on time delay between left and right audio channels
# Inputs:
#   left (np.array) – Left channel waveform
#   right (np.array) – Right channel waveform
#   sr (int) – Sample rate
#   mic_distance (float) – Distance between microphones (in meters), default is 0.2 m
# Output:
#   float – Estimated azimuth angle (degrees), or 0.0 if estimation fails
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')  # Cross-correlation to measure delay
    delay = np.argmax(corr) - (len(right) - 1)  # Time delay between signals (in samples)
    time_diff = delay / sr  # Convert delay to seconds
    try:
        # Use the speed of sound (343 m/s) to calculate angle from time difference
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)  # Convert radians to degrees
    except:
        return 0.0  # If math fails (like invalid arcsin input), return 0

# === Function: extract_features ===
# Purpose: Extracts MFCC, spectral, and directional features from a stereo audio file
# Input:
#   file_path (str) – Path to stereo audio file (.wav)
# Output:
#   np.array – Combined feature vector (or None if error)
def extract_features(file_path):
    try:
        # Load audio file with both channels (mono=False)
        y, sr = librosa.load(file_path, sr=None, mono=False)

        # Check if audio is stereo (2 channels)
        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None  # Skip file if it's mono

        # Split stereo signal into left and right
        left, right = y[0], y[1]

        # === Sub-function: get_audio_features ===
        # Extracts multiple features from a single audio channel
        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # MFCCs (13 coefficients)
            mfcc_mean = np.mean(mfcc.T, axis=0)  # Mean of MFCCs across time
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))  # Spectral centroid
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))  # Spectral rolloff
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))  # Zero crossing rate
            rms = np.mean(librosa.feature.rms(y=signal))  # Root Mean Square energy
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))  # Bandwidth
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))  # Chroma STFT (12 pitch classes)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))  # Tonnetz (tonal centroid)

            # Combine all features into a single 1D vector
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])

        # Extract features for both channels
        left_feat = get_audio_features(left)
        right_feat = get_audio_features(right)

        # Estimate azimuth angle
        angle = estimate_direction(left, right, sr)

        # Combine all features into one feature vector
        return np.hstack([left_feat, right_feat, angle])

    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None  # Return None if any error occurs during processing

# === Load label sheet (Excel) and extract features ===

df = pd.read_excel(EXCEL_PATH)  # Load Excel file with filename and intent columns
features, labels = [], []  # Lists to store features and corresponding labels

print("🔍 Extracting features...")
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])  # Construct full path to audio file
    feat = extract_features(path)  # Extract features
    if feat is not None:
        features.append(feat)  # Append feature vector
        labels.append(row['intent'])  # Append corresponding label
print(f"✅ Extracted {len(features)} samples.")  # Show number of successful extractions

# === Convert features and labels to arrays ===

X = np.array(features)  # Feature matrix
y = np.array(labels)    # Target labels

# === Normalize features ===

scaler = StandardScaler()  # Standard scaler to normalize features
X_scaled = scaler.fit_transform(X)  # Apply normalization

# === Split dataset into training and testing sets ===

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Train model using GridSearchCV ===

print("🔧 Training model with GridSearch...")
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [None, 20, 40],  # Tree depth
    'min_samples_split': [2, 5],  # Min samples required to split
    'min_samples_leaf': [1, 2],  # Min samples in leaf node
    'class_weight': [None, 'balanced']  # Deal with imbalanced classes
}

# Perform grid search with 3-fold cross-validation
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1  # Use all available CPU cores
)
grid.fit(X_train, y_train)  # Train model on training data

# === Evaluate best model on test set ===

best_model = grid.best_estimator_  # Get best model from grid search

print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)  # Predict on test set
print(classification_report(y_test, y_pred))  # Show precision, recall, F1-score
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")  # Print accuracy

# === Save model and scaler ===

joblib.dump(best_model, 'horn_intent_with_sound_localization_model.pkl')  # Save model
joblib.dump(scaler, 'feature_scaler.pkl')  # Save scaler
print("💾 Model saved: horn_intent_with_sound_localization_model.pkl")
