# ===== Import necessary libraries =====
import os  # For file and folder path operations
import pandas as pd  # For reading Excel sheets and working with tabular data
import librosa  # For audio loading and feature extraction
import numpy as np  # For handling numerical data and operations
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # The chosen ML model for horn intent recognition
from sklearn.preprocessing import StandardScaler  # For feature normalization
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance
import joblib  # For saving and loading the trained model and scaler

# ===== Define paths =====
AUDIO_DIR = 'audio'  # Directory containing .wav audio files
EXCEL_PATH = 'horn_intents.xlsx'  # Path to Excel file containing labels and filenames

# ===== Function: estimate_direction =====
# Calculates the angle from which the horn sound is coming based on time delay between two microphones
# Inputs:
#   - left: numpy array of samples from left channel
#   - right: numpy array of samples from right channel
#   - sr: sampling rate
#   - mic_distance: distance between the two microphones (default 0.2 meters)
# Output:
#   - angle in degrees (float)
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')  # Cross-correlation between left and right signals
    delay = np.argmax(corr) - (len(right) - 1)  # Delay in samples
    time_diff = delay / sr  # Time delay in seconds

    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))  # Calculate angle in radians
        return np.degrees(angle_rad)  # Convert to degrees
    except:
        return 0.0  # Return 0 if angle can't be computed

# ===== Function: extract_features =====
# Extracts features from stereo audio file and appends estimated direction
# Inputs:
#   - file_path: string path to a stereo .wav file
# Output:
#   - A numpy array of combined left + right features + direction, or None on error
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)  # Load stereo audio file

        if y.ndim < 2:  # Ensure the file is stereo
            print(f"❗ File not stereo: {file_path}")
            return None

        left, right = y[0], y[1]  # Separate left and right channels

        # === Inner Function: get_audio_features ===
        # Extracts a series of features from a single channel
        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extract MFCCs
            mfcc_mean = np.mean(mfcc.T, axis=0)  # Average over time
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))  # Centroid
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))  # Spectral roll-off
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))  # Zero crossing rate
            rms = np.mean(librosa.feature.rms(y=signal))  # Root Mean Square energy
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))  # Bandwidth
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))  # Chroma vector
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))  # Tonnetz
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])  # Combine all

        left_feat = get_audio_features(left)   # Features from left mic
        right_feat = get_audio_features(right) # Features from right mic

        angle = estimate_direction(left, right, sr)  # Estimate direction of horn

        return np.hstack([left_feat, right_feat, angle])  # Combine all into one feature vector
    except Exception as e:
        print(f"Error: {file_path} -> {e}")  # Print error if extraction fails
        return None

# ===== Read the Excel file containing filenames and labels =====
df = pd.read_excel(EXCEL_PATH)  # Load Excel file into a DataFrame

# ===== Lists to store features and their corresponding labels =====
features = []  # List to hold feature arrays
labels = []    # List to hold intent labels

print("🔍 Extracting features...")
# ===== Loop over all rows in the Excel file =====
for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row['filename'])  # Construct full path to audio file
    feat = extract_features(path)  # Extract features
    if feat is not None:
        features.append(feat)         # Append valid features
        labels.append(row['intent'])  # Append label (intent type)

print(f"✅ Extracted {len(features)} samples.")  # Display number of successful extractions

# ===== Convert feature and label lists to NumPy arrays =====
X = np.array(features)  # Feature matrix
y = np.array(labels)    # Label vector

# ===== Normalize the features (zero mean, unit variance) =====
scaler = StandardScaler()        # Create a scaler instance
X_scaled = scaler.fit_transform(X)  # Fit and transform the feature matrix

# ===== Split the data into training and test sets (80/20) =====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===== Define parameter grid for hyperparameter tuning =====
print("🔧 Training model with GridSearch...")
param_grid = {
    'n_estimators': [100, 200],         # Number of trees in the forest
    'max_depth': [None, 20, 40],        # Max depth of the trees
    'min_samples_split': [2, 5],        # Min samples required to split an internal node
    'min_samples_leaf': [1, 2],         # Min samples required to be at a leaf node
    'class_weight': [None, 'balanced']  # Deal with class imbalance
}

# ===== Create and train the model using GridSearchCV =====
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),  # Use Random Forest
    param_grid,  # Pass the hyperparameter grid
    cv=3,        # 3-fold cross-validation
    n_jobs=-1    # Use all processors available
)
grid.fit(X_train, y_train)  # Train model using training set

# ===== Use the best estimator from the Grid Search =====
best_model = grid.best_estimator_  # Best model configuration

# ===== Evaluate performance on test data =====
print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)  # Predict on test set
print(classification_report(y_test, y_pred))  # Print classification metrics
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")  # Print accuracy percentage

# ===== Save the trained model and scaler to disk =====
joblib.dump(best_model, 'horn_intent_model.pkl')  # Save model to file
joblib.dump(scaler, 'scaler.pkl')                 # Save scaler for future use
print("💾 Model saved: horn_intent_model.pkl")  # Notify user
