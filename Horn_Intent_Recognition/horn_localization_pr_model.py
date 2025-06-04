# Importing necessary Python libraries

import os                      # Used to interact with the operating system, like checking if a file exists
import joblib                  # Used to load the saved machine learning model
import librosa                 # A library to process audio files, commonly used for music and sound analysis
import numpy as np             # Used for numerical operations, like working with arrays and mathematical functions
import matplotlib.pyplot as plt  # Used to create graphs and visualizations

# Configuration: Setting the model file path and audio file path
MODEL_PATH = 'horn_intent_model_with_angle.pkl'  # Path to the trained machine learning model
AUDIO_FILE = 'Horn4.wav'  # Audio file to analyze (must be stereo — two audio channels)

# Load the machine learning model
print("🔍 Loading model...")
model = joblib.load(MODEL_PATH)  # Load the pre-trained model from the file

# Function to estimate the direction (angle) from which the horn sound is coming
def estimate_direction(left, right, sr, mic_distance=0.2):
    # Calculate the similarity (correlation) between left and right channels
    corr = np.correlate(left, right, 'full')
    
    # Find the point of maximum similarity (delay between two microphones)
    delay = np.argmax(corr) - (len(right) - 1)
    
    # Calculate time difference based on sampling rate
    time_diff = delay / sr
    
    try:
        # Estimate the angle of arrival using the time difference, speed of sound, and mic distance
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))  # 343 m/s is speed of sound
        return np.degrees(angle_rad)  # Convert from radians to degrees
    except:
        return 0.0  # If any error occurs (like invalid angle), return 0 degrees

# Function to extract features from the audio file (used for prediction)
def extract_features(file_path):
    try:
        # Load the audio file. Set mono=False to keep stereo channels (left & right)
        y, sr = librosa.load(file_path, sr=None, mono=False)

        # Check if the file is actually stereo
        if y.ndim < 2:
            print("❗ Stereo file required.")
            return None, None  # Return nothing if it's not stereo

        # Separate left and right audio channels
        left, right = y[0], y[1]

        # Extract MFCC features (representing timbre or sound quality)
        mfcc = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Average across time

        # Extract spectral centroid (represents "brightness" of the sound)
        centroid = np.mean(librosa.feature.spectral_centroid(y=left, sr=sr))

        # Extract spectral rolloff (indicates frequency below which a percentage of total energy is contained)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=left, sr=sr))

        # Extract zero crossing rate (how often the signal changes sign, indicates noisiness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(left))

        # Extract RMS (Root Mean Square) energy (represents loudness)
        rms = np.mean(librosa.feature.rms(y=left))

        # Estimate direction (angle of arrival)
        angle = estimate_direction(left, right, sr)

        # Combine all features into a single list (vector)
        combined = np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, angle])

        # Return the feature vector and the angle separately
        return combined, angle

    except Exception as e:
        # In case of error, show the error message
        print(f"❌ Error: {e}")
        return None, None

# Now start prediction if the audio file exists
if not os.path.exists(AUDIO_FILE):
    print(f"❌ File not found: {AUDIO_FILE}")  # Warn if the file does not exist
else:
    print(f"🎧 Analyzing {AUDIO_FILE}...")  # Tell the user which file is being analyzed

    # Extract features and angle from the audio
    features, angle = extract_features(AUDIO_FILE)

    if features is not None:
        # Reshape the features for the model (models expect 2D input)
        features = features.reshape(1, -1)

        # Use the model to make a prediction based on extracted features
        prediction = model.predict(features)[0]

        # Display the predicted intent (like 'warning', 'presence', etc.)
        print(f"\n🔊 Predicted Intent: **{prediction}**")

        # Display the estimated direction of the sound in degrees
        print(f"📐 Estimated Direction: {angle:.2f}°")

        # Create a polar plot (circular graph) to visually show the direction
        plt.figure()
        plt.polar([0, np.radians(angle)], [0, 1], marker='o')  # Convert angle to radians
        plt.title(f"Direction: {angle:.2f}°")  # Add title with angle
        plt.show()  # Show the plot on the screen

    else:
        # If feature extraction failed, notify the user
        print("⚠️ Feature extraction failed.")
