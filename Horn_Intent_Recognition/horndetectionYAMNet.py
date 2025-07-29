import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING logs

import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import librosa
import tensorflow as tf
import yamnet_model
import params

model = yamnet_model.yamnet_frames_model(params.Params())
model.load_weights('yamnet.h5')

class_names = yamnet_model.class_names('yamnet_class_map.csv')


# Define all relevant horn indices (including French horn)
horn_indices = [181, 302, 312, 325, 395]

# Folder containing your wav files
input_folder = "input"
wav_files = glob.glob(os.path.join(input_folder, "*.wav"))

# Confidence threshold
horn_threshold = 0.1  # Adjust as needed

# YAMNet uses ~0.48s hop and ~0.96s window
frame_hop_seconds = 0.48

for wav_file in wav_files:
    print(f"\nProcessing file: {wav_file}")

    # Load audio
    waveform, sr = librosa.load(wav_file, sr=16000)

    # Run inference
    scores, embeddings, spectrogram = model(waveform)

    # Convert to numpy
    scores_np = scores.numpy()

    # === Frame-based horn detection ===
    horn_detections = []
    for frame_idx, frame_scores in enumerate(scores_np):
        # Get max horn confidence for this frame
        horn_conf = max([frame_scores[idx] for idx in horn_indices])
        if horn_conf >= horn_threshold:
            # Compute timestamp
            start_time = frame_idx * frame_hop_seconds
            end_time = start_time + 0.96  # approximate window length
            horn_detections.append((start_time, end_time, horn_conf))

    # Print results
    if horn_detections:
        print("✅ Horn detected in the following segments:")
        for start, end, conf in horn_detections:
            print(f"  {start:.2f}s - {end:.2f}s (confidence {conf:.3f})")
    else:
        print("❌ No horn detected in any frame.")

    # === Optional: Print top 5 average predictions over all frames
    mean_scores = np.mean(scores_np, axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:5]
    print("\nTop 5 predictions (overall):")
    for i in top_indices:
        print(f"  {class_names[i]}: {mean_scores[i]:.3f}")
