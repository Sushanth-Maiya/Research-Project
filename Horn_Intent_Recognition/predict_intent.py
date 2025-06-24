# ===== Import necessary libraries =====
import os                       # Used to interact with the operating system, like file management
import glob                     # Used to search for files matching a specified pattern
import joblib                   # Used to load the trained machine learning model and scaler
import librosa                  # Library used for audio processing
import numpy as np              # Library for numerical operations like arrays and mathematical functions
import matplotlib.pyplot as plt  # Used for plotting the direction on a polar chart
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

# ===== Load model and scaler =====
print("🔍 Loading model and scaler...")                            # Inform that model and scaler are being loaded
model = joblib.load('horn_intent_model.pkl')                    # Load the pre-trained horn intent classification model
scaler = joblib.load('scaler.pkl')                              # Load the scaler used for feature normalization

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
    tau = gcc_phat(left, right, fs=sr)                           # Calculate time delay using GCC-PHAT
    try:
        angle_rad = np.arcsin(np.clip(tau * 343 / mic_distance, -1, 1))  # Calculate angle in radians using speed of sound (343 m/s)
        return np.degrees(angle_rad)                             # Convert radians to degrees and return
    except:
        return 0.0                                               # Return 0 if calculation fails

# ===== Function: extract_features =====
# Function Name: extract_features
# Purpose: Extracts audio features from both left and right mic input files and estimates the direction angle
# Inputs:
#   - left_path: File path for the left microphone audio (.wav)
#   - right_path: File path for the right microphone audio (.wav)
# Output:
#   - Tuple containing the combined feature vector and the estimated angle


def extract_features(left_path, right_path):
    try:
        # Load stereo audio files from both microphones
        y_left, sr_left = librosa.load(left_path, sr=None, mono=False)   # Load left audio file as stereo
        y_right, sr_right = librosa.load(right_path, sr=None, mono=False) # Load right audio file as stereo

        # Extract first channel from stereo files (mono signal from each mic)
        if y_left.ndim == 2:
            left = y_left[0]   # Select first channel if stereo
        else:
            left = y_left      # Use directly if mono

        if y_right.ndim == 2:
            right = y_right[0] # Select first channel if stereo
        else:
            right = y_right    # Use directly if mono

        sr = sr_left   # Use sampling rate from the left mic (both are same)

        # === Internal function to extract features from audio signal ===
        # Function Name: get_audio_features
        # Purpose: Extracts various audio features from the given audio signal
        # Input:
        #   - signal: Audio signal array
        # Output:
        #   - Combined feature vector containing MFCCs, spectral features, and tonal features

        def get_audio_features(signal):
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)            # Extract 13 MFCC coefficients
            mfcc_mean = np.mean(mfcc.T, axis=0)                                # Mean of MFCCs over time
            centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))  # Mean spectral centroid
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))    # Mean spectral rolloff
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))            # Mean zero-crossing rate
            rms = np.mean(librosa.feature.rms(y=signal))                           # Mean RMS energy
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))# Mean spectral bandwidth
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))          # Mean chroma feature
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr))  # Mean tonnetz feature
            return np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, bandwidth, chroma, tonnetz])  # Combine all features

        left_feat = get_audio_features(left)     # Extract features from left mic signal
        right_feat = get_audio_features(right)   # Extract features from right mic signal

        angle = estimate_direction(left, right, sr)  # Estimate direction of the horn sound

        return np.hstack([left_feat, right_feat, angle]), angle  # Combine all features and angle as final feature vector
    except Exception as e:
        print(f"Error extracting features: {e}")  # Print error if extraction fails
        return None, None

# ===== Prediction Process =====
LEFT_AUDIO = glob.glob('input/left/*.wav')      # Get all left mic audio files from folder
RIGHT_AUDIO = glob.glob('input/right/*.wav')    # Get all right mic audio files from folder

if not LEFT_AUDIO or not RIGHT_AUDIO:            # Check if audio files exist in both folders
    print("❌ No audio files found in input/left/ or input/right/ directory.")
else:
    left_file = LEFT_AUDIO[0]                    # Select the first left mic audio file
    right_file = RIGHT_AUDIO[0]                  # Select the first right mic audio file
    print(f"🎧 Left Input File: {left_file}")
    print(f"🎧 Right Input File: {right_file}")

    features, angle = extract_features(left_file, right_file)   # Extract features and direction angle

    if features is not None:
        features_scaled = scaler.transform([features])          # Scale the extracted features using the loaded scaler
        predicted_intent = model.predict(features_scaled)[0]    # Predict the horn intent using the trained model

        print(f"\n🔊 Predicted Horn Intent: {predicted_intent}")
        print(f"📐 Estimated Direction: {angle:.2f}°")              # Print the estimated angle in degrees

        # # ===== Plot the direction on a polar chart =====
        # plt.figure()                                     # Create a new figure
        # ax = plt.subplot(111, polar=True)               # Create a polar subplot (for circular angle plotting)
        # ax.plot([0, np.radians(angle)], [0, 1],         # Plot a line from center to angle direction
        #         marker='o', color='red', linewidth=2)
        # ax.set_title(f"Direction: {angle:.2f}°", va='bottom')  # Set title showing numeric angle
        # plt.show()                                      # Display the polar plot

        # Classify direction based on the estimated angle
        if angle < -30:
            direction = "Right Back"
        elif -30 <= angle < -10:
            direction = "Right"
        elif -10 <= angle <= 10:
            direction = "Center Back"
        elif 10 < angle <= 30:
            direction = "Left"
        else:
            direction = "Left Back"

        print(f"📍 Estimated Horn Direction: {direction} (Angle: {angle:.2f}°)")  # Print final direction label and angle

    else:
        print("❗ Feature extraction failed.")