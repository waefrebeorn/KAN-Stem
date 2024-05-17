import numpy as np
import librosa


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def separate_stems(audio):
    # Placeholder for KAN implementation
    return {"vocals": audio, "accompaniment": audio}


def save_stems(stems, sr, output_dir):
    for stem_name, stem_audio in stems.items():
        librosa.output.write_wav(f"{output_dir}/{stem_name}.wav", stem_audio, sr)
