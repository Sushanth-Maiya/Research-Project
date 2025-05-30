import librosa

def is_stereo(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim == 1:
            print(f"🔍 {file_path} is MONO")
            return False
        elif y.ndim == 2 and y.shape[0] == 2:
            print(f"🔍 {file_path} is STEREO type")
            return True
        else:
            print(f"⚠️ Unusual channel format in {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

# Example usage:
AUDIO_FILE = 'Horn1.wav'  # <- Replace with your test file
is_stereo(AUDIO_FILE)
