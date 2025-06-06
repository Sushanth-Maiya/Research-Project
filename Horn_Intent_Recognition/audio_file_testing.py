import os
import librosa
import numpy as np

# 📂 Set your dataset directory here
AUDIO_DIR = "audio"  # change this if your folder is named differently

# 🔎 Scan all .wav files
def check_stereo_files(audio_dir):
    results = []

    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(".wav"):
            path = os.path.join(audio_dir, filename)
            try:
                y, sr = librosa.load(path, sr=None, mono=False)

                if y.ndim == 1:
                    status = "❌ Mono"
                elif np.allclose(y[0], y[1], atol=1e-5):
                    status = "⚠️ Stereo (identical channels)"
                else:
                    status = "✅ Stereo (different channels)"
            except Exception as e:
                status = f"❗ Error: {e}"

            results.append((filename, status))

    return results

# 🖨️ Run the check and print results
results = check_stereo_files(AUDIO_DIR)
for fname, status in results:
    print(f"{fname}: {status}")

