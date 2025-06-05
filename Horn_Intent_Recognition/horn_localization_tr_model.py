# Import necessary tools (called libraries) that help us with different tasks
import os  # Helps to work with files and folders on the computer
import pandas as pd  # Used to read and manage Excel files (data tables)
import librosa  # A library that helps read and analyze sound files (audio)
import numpy as np  # Helps do math calculations easily
from sklearn.model_selection import train_test_split  # Helps split data into training and testing groups
from sklearn.ensemble import RandomForestClassifier  # A tool to create a machine learning model for classification
from sklearn.metrics import classification_report, accuracy_score  # Helps us see how well the model is doing
import joblib  # Used to save the model so we can use it later

# Set the location of the audio files and the Excel file
AUDIO_DIR = 'audio'  # The folder where all horn sound files are stored
EXCEL_PATH = 'horn_intents.xlsx'  # The Excel file that contains the filename and intent label for each audio

# 🧠 Function to guess from which direction the sound came
def estimate_direction(left, right, sr, mic_distance=0.2):
    # left and right are the sound signals from left and right microphones
    # sr is the sample rate (how many pieces of audio are taken per second)
    # mic_distance is the distance between left and right microphones (in meters, here 20 cm)

    # Compare left and right to find time delay (cross-correlation)
    corr = np.correlate(left, right, 'full')  # Check how similar left and right sounds are, with different time shifts
    delay = np.argmax(corr) - (len(right) - 1)  # Find the time delay between left and right
    time_diff = delay / sr  # Convert delay (in samples) to actual time (in seconds)
    
    try:
        # Calculate the angle (direction) the sound came from using physics
        angle_rad = np.arcsin(np.clip(time_diff * 343 / mic_distance, -1, 1))
        return np.degrees(angle_rad)  # Convert angle from radians to degrees
    except:
        # If something goes wrong (like math error), just return 0 (assume front)
        return 0.0

# 🎵 Function to take an audio file and turn it into useful numbers (features) for the machine
def extract_features(file_path):
    try:
        # Load the sound file from the given location
        # mono=False means we want both left and right signals (stereo)
        y, sr = librosa.load(file_path, sr=None, mono=False)
        
        # If it's not stereo (only one sound channel), skip it
        if y.ndim < 2:
            print(f"❗ File not stereo: {file_path}")
            return None

        # Separate the left and right microphone recordings
        left, right = y[0], y[1]

        # Extract MFCC: numbers that capture the shape and style of the sound
        mfcc = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=13)  # Get 13 MFCC features
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Take the average of each MFCC over time

        # Get other sound features:
        centroid = np.mean(librosa.feature.spectral_centroid(y=left, sr=sr))  # Where the "center" of the sound is
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=left, sr=sr))  # Point where high frequencies drop off
        zcr = np.mean(librosa.feature.zero_crossing_rate(left))  # How often the signal crosses zero (measures noisiness)
        rms = np.mean(librosa.feature.rms(y=left))  # Measures loudness

        # Estimate where the sound came from (angle)
        angle = estimate_direction(left, right, sr)

        # Return all features combined into one list
        return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, angle])
    except Exception as e:
        # If anything goes wrong, print an error and skip this file
        print(f"Error: {file_path} -> {e}")
        return None

# 📋 Read the Excel file that tells us which audio file has what kind of horn sound
df = pd.read_excel(EXCEL_PATH)  # Read the Excel file into a table (called a DataFrame)
features = []  # This will store the numbers extracted from audio files
labels = []  # This will store the correct answers (intent type) for each audio

print("🔍 Extracting features...")
# Go through each row in the Excel file, one by one
for _, row in df.iterrows():
    # Get the full path to the audio file
    path = os.path.join(AUDIO_DIR, row['filename'])

    # Extract sound features from this file
    feat = extract_features(path)

    # If features were successfully extracted, add them to the list
    if feat is not None:
        features.append(feat)  # Store the extracted numbers
        labels.append(row['intent'])  # Store the correct label for this audio

# Show how many audio files we successfully processed
print(f"✅ Extracted {len(features)} samples.")

# 🤖 Split the data into two groups: training and testing
# 80% of data is used to train the machine, 20% to test how well it learned
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 🎯 Create a Random Forest Classifier – a machine that learns from many decision trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Use 100 trees
clf.fit(X_train, y_train)  # Train the model using the training data

# 🧪 Test the trained model on new, unseen data
print("\n📊 Classification Report:")
y_pred = clf.predict(X_test)  # Ask the model to guess the intent for test data

# Print a detailed report of how well the model did
print(classification_report(y_test, y_pred))  # Shows accuracy, precision, etc.
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")  # Show simple percentage of correct guesses

# 💾 Save the trained model to a file so we can use it later without training again
joblib.dump(clf, 'horn_intent_model_with_angle.pkl')
print("💾 Model saved: horn_intent_model_with_angle.pkl")
