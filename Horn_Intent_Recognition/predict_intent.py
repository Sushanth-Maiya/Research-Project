# ===== Import necessary libraries =====
import os                # For file system operations like checking if files exist
import glob              # For finding file paths using wildcard patterns
import joblib            # For loading saved machine learning models and scalers
import librosa           # For audio processing and feature extraction
import numpy as np       # For numerical computations and handling arrays
import matplotlib.pyplot as plt  # For plotting the angle in a polar chart

# ===== Define paths for the trained model, scaler, and input audio =====
MODEL_PATH = 'horn_intent_model.pkl'         # File path to the trained Random Forest model
SCALER_PATH = 'scaler.pkl'                   # File path to the trained feature scaler (StandardScaler)
LEFT_AUDIO = glob.glob('input/left/*.wav')   # Finds all .wav files in the 'input/left/' folder (left mic audio)
RIGHT_AUDIO = glob.glob('input/right/*.wav') # Finds all .wav files in the 'input/right/' folder (right mic audio)

# ===== Load the trained model and scaler from disk =====
print("🔍 Loading model and scaler...")
model = joblib.load(MODEL_PATH)   # Loads the saved trained ML model from disk
scaler = joblib.load(SCALER_PATH) # Loads the saved StandardScaler used during model training

# ===== Function: estimate_direction =====
# This function estimates the direction of the horn sound using the time delay between left and right microphones
# Inputs:
#   - left: numpy array of audio samples from the left microphone
#   - right: numpy array of audio samples from the right microphone
#   - sr: sampling rate of the audio (samples per second)
#   - mic_distance: distance between the two microphones (default is 0.2 meters)
# Output:
#   - Estimated angle of the sound source in degrees
def estimate_direction(left, right, sr, mic_distance=0.2):
    corr = np.correlate(left, right, 'full')  # Cross-correlate left and right channels to find similarity at various delays
    delay = np.argmax(corr) - (len(right) - 1)  # Delay is index of peak correlation minus center point
    time_diff = delay / sr  # Converts delay in samples to time in seconds

    try:
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))  # Uses the time difference and speed of sound to estimate angle in radians
        return np.degrees(angle_rad)  # Converts radians to degrees
    except:
        return 0.0  # Returns 0.0 if arcsin fails due to invalid input (e.g., out of domain)

# ===== Function: extract_features =====
# This function extracts audio features from a pair of left and right audio files, and also estimates the direction
# Inputs:
#   - left_path: file path to left mic audio (.wav)
#   - right_path: file path to right mic audio (.wav)
# Outputs:
#   - features: combined array of features from left + right + angle
#   - angle: estimated angle of sound source
def extract_features(left_path, right_path):
    try:
        y_left, sr_left = librosa.load(left_path, sr=None, mono=False)   # Load left audio file in stereo if available
        y_right, sr_right = librosa.load(right_path, sr=None, mono=False) # Load right audio file in stereo if available

        # If stereo, use only first channel. If mono, use as-is.
        if y_left.ndim == 2:
            left = y_left[0]  # Select left channel
        else:
            left = y_left     # Use mono channel directly

        if y_right.ndim == 2:
            right = y_right[0]  # Select left channel (right mic file)
        else:
            right = y_right     # Use mono channel directly

        sr = sr_left  # Assume both audio files have the same sampling rate

        # === Inner function: get_audio_features ===
        # Extracts various audio features from a given signal (MFCCs, ZCR, Centroid, etc.)
        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)                 # Extract 13 MFCCs
            mfcc_mean = np.mean(mfcc.T, axis=0)                                     # Compute mean of MFCCs over time
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)) # Measure center of mass of frequency spectrum
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))   # Frequency where roll-off occurs
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))            # Rate of zero crossings (frequency estimate)
            rms = np.mean(librosa.feature.rms(y=signal))                           # Energy of the signal
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)) # Width of the frequency spectrum
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))         # Chroma: pitch content
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)) # Tonal centroid features
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz]) # Combine all features

        # Get feature vectors for both channels
        left_feat = get_audio_features(left)   # Extract features from left mic
        right_feat = get_audio_features(right) # Extract features from right mic

        angle = estimate_direction(left, right, sr)  # Estimate direction using time difference

        features = np.hstack([left_feat, right_feat, angle])  # Combine all features and direction angle into one vector
        return features, angle  # Return combined feature vector and angle
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")  # Print any error during extraction
        return None, None  # Return None if failed

# ===== Prediction block (main script execution) =====
if not LEFT_AUDIO or not RIGHT_AUDIO:
    print("❌ No audio files found in input/left/ or input/right/")  # Print error if audio files are missing
else:
    left_file = LEFT_AUDIO[0]   # Select first left mic .wav file
    right_file = RIGHT_AUDIO[0] # Select first right mic .wav file
    print(f"🎧 Left: {left_file}")   # Print which left file is used
    print(f"🎧 Right: {right_file}") # Print which right file is used

    features, angle = extract_features(left_file, right_file)  # Extract audio features and direction

    if features is not None:
        features_scaled = scaler.transform([features])  # Normalize features using trained scaler
        prediction = model.predict(features_scaled)[0]  # Predict the horn intent class (returns label)

        print(f"\n🔊 Predicted Intent: **{prediction}**")         # Output predicted class
        print(f"📐 Estimated Direction: {angle:.2f}°")           # Output estimated direction in degrees

        # ===== Plot the direction on a polar chart (circular direction indicator) =====
        plt.figure()                                     # Create a new figure
        ax = plt.subplot(111, polar=True)               # Create a polar subplot
        ax.plot([0, np.radians(angle)], [0, 1],         # Plot a line from center to angle direction
                marker='o', color='red', linewidth=2)
        ax.set_title(f"Direction: {angle:.2f}°", va='bottom')  # Set title showing numeric angle
        plt.show()                                      # Display the plot
    else:
        print("⚠️ Could not extract features.")  # Print warning if feature extraction failed
