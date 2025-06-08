 # === Import necessary libraries ===
import os  # For checking if the file exists
import joblib  # For loading the saved ML model and scaler
import librosa  # For audio processing and feature extraction
import numpy as np  # For numerical operations like arrays, mean, stacking
import matplotlib.pyplot as plt  # For plotting the direction using polar coordinates

# === File paths ===
MODEL_PATH = 'horn_intent_with_sound_localization_model.pkl'  # Path to saved trained model
SCALER_PATH = 'feature_scaler.pkl'  # Path to saved StandardScaler object
AUDIO_FILE = 'Horn4.wav'  # Input stereo audio file to be analyzed

# === Load pre-trained model and feature scaler ===
print("🔍 Loading model and scaler...")
model = joblib.load(MODEL_PATH)  # Load the machine learning model
scaler = joblib.load(SCALER_PATH)  # Load the scaler to normalize new input features

# === Function: estimate_direction ===
# This function estimates the angle (direction) of the sound source based on stereo channels
# Inputs:
#   left, right: numpy arrays of left and right audio channels
#   sr: sampling rate
#   mic_distance: distance between the two microphones (default 0.2 meters)
# Output:
#   Estimated angle in degrees
def estimate_direction(left, right, sr, mic_distance=0.2):
    # Cross-correlate left and right channels to find delay
    corr = np.correlate(left, right, 'full')  # 'full' mode gives complete correlation curve
    delay = np.argmax(corr) - (len(right) - 1)  # Time delay = index of peak - midpoint
    time_diff = delay / sr  # Convert delay in samples to time (seconds)

    try:
        # Estimate angle using speed of sound (343 m/s) and mic distance
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))  # Clip ensures input is in [-1, 1]
        return np.degrees(angle_rad)  # Convert angle from radians to degrees
    except:
        return 0.0  # If invalid (e.g., arcsin out of range), return 0 as fallback

# === Function: extract_features ===
# This function extracts audio features from a stereo file and estimates the direction
# Input:
#   file_path: path to the stereo .wav file
# Output:
#   full_features: combined feature vector from left + right + direction
#   angle: estimated angle of sound source
def extract_features(file_path):
    try:
        # Load stereo audio (mono=False ensures both left and right channels are returned)
        y, sr = librosa.load(file_path, sr=None, mono=False)

        # Validate that audio has two channels
        if y.ndim < 2:
            print("❗ Stereo file required.")  # If only one channel, abort
            return None, None

        # Separate left and right audio channels
        left, right = y[0], y[1]

        # === Sub-function: get_audio_features ===
        # Extracts relevant audio features from one channel
        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # MFCC coefficients
            mfcc_mean = np.mean(mfcc.T, axis=0)  # Mean of MFCC over time

            # Extract spectral and energy-based features
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))  # Frequency center
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))  # Where energy "rolls off"
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))  # Rate of zero crossings
            rms = np.mean(librosa.feature.rms(y=signal))  # Root mean square energy
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))  # Bandwidth of signal
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))  # Chroma (pitch class) features
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))  # Tonal centroid

            # Combine all features into a single vector
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])

        # Extract features separately for left and right channels
        left_feat = get_audio_features(left)
        right_feat = get_audio_features(right)

        # Estimate direction (angle) of sound source
        angle = estimate_direction(left, right, sr)

        # Combine all features into one long vector: [left features | right features | angle]
        full_features = np.hstack([left_feat, right_feat, angle])

        return full_features, angle  # Return the final feature vector and angle
    except Exception as e:
        # Catch and print any errors (e.g., corrupted audio)
        print(f"❌ Error: {e}")
        return None, None  # Return failure signal

# === Main Prediction Block ===
# This part checks for file existence, extracts features, scales them,
# makes prediction using the model, and visualizes direction
if not os.path.exists(AUDIO_FILE):
    # If the audio file does not exist, alert the user
    print(f"❌ File not found: {AUDIO_FILE}")
else:
    # If file exists, continue with prediction
    print(f"🎧 Analyzing {AUDIO_FILE}...")

    # Extract features and direction from audio file
    features, angle = extract_features(AUDIO_FILE)

    # If features were successfully extracted
    if features is not None:
        # Scale (normalize) the features using the previously saved scaler
        features_scaled = scaler.transform([features])  # Must pass 2D array to scaler

        # Predict the horn intent class using the trained model
        prediction = model.predict(features_scaled)[0]  # Get the predicted label (string or int)

        # Display predicted intent and direction
        print(f"\n🔊 Predicted Intent: **{prediction}**")
        print(f"📐 Estimated Direction: {angle:.2f}°")

        # ==== Plot the angle on a polar plot (circular graph) ====
        plt.figure()  # Create new plot
        plt.polar([0, np.radians(angle)], [0, 1], marker='o', color='red')  # Plot arrow to angle
        plt.title(f"Direction: {angle:.2f}°")  # Title shows numeric direction
        plt.show()  # Display the plot

    else:
        # If feature extraction failed
        print("⚠️ Feature extraction failed.")
