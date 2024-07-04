import os
import random
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import warnings
import librosa

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def ensure_mono(waveform):
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform

def standardize_length(waveform, sample_rate, target_length):
    target_samples = target_length * sample_rate
    waveform_length = len(waveform)

    if waveform_length >= target_samples:
        return waveform[:target_samples]
    else:
        return np.pad(waveform, (0, target_samples - waveform_length), 'constant')

def resample_audio(waveform, orig_sr, target_sr):
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform

def save_as_ogg(waveform, sample_rate, file_path):
    temp_wav_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace('.ogg', '.wav'))
    sf.write(temp_wav_path, waveform, sample_rate)
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(file_path, format='ogg')
    os.remove(temp_wav_path)

def process_and_save_file(input_file, output_file, target_sr, target_length):
    waveform, sr = sf.read(input_file, dtype='float32')
    waveform = ensure_mono(waveform)
    waveform = resample_audio(waveform, sr, target_sr)
    waveform = standardize_length(waveform, target_sr, target_length)
    save_as_ogg(waveform, target_sr, output_file)

def combine_and_shuffle_stems(input_dir, output_dir, num_examples=100, target_length=60, sample_rate=44100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stem_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    file_lists = {}
    for stem_dir in stem_dirs:
        files = [f for f in os.listdir(os.path.join(input_dir, stem_dir)) if f.endswith(('.wav', '.ogg', '.flac'))]
        file_lists[stem_dir] = files

    min_files = min(len(files) for files in file_lists.values())
    print(f"Found at least {min_files} audio files in each stem directory")
    
    data = []
    targets = {stem: [] for stem in file_lists.keys()}

    for i in range(num_examples):
        combined_waveform = None
        max_length = target_length * sample_rate  # Target length in samples

        for stem_name, files in file_lists.items():
            file = random.choice(files)  # Randomly choose a file from each stem directory
            input_file_path = os.path.join(input_dir, stem_name, file)
            output_file_path = os.path.join(output_dir, f'{stem_name}_{i+1:04d}.ogg')

            process_and_save_file(input_file_path, output_file_path, sample_rate, target_length)
            waveform, _ = sf.read(output_file_path, dtype='float32')

            if combined_waveform is None:
                combined_waveform = np.zeros_like(waveform)
            
            combined_waveform[:len(waveform)] += waveform

        combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))
        combined_output_file = os.path.join(output_dir, f'example_{i+1:04d}.ogg')
        save_as_ogg(combined_waveform, sample_rate, combined_output_file)

        print(f"Generated example {i+1}")

def organize_and_prepare_dataset(input_dir, output_dir, num_examples, target_length, sample_rate):
    combine_and_shuffle_stems(input_dir, output_dir, num_examples, target_length, sample_rate)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

def organize_and_prepare_dataset_gradio(input_dir, output_dir, num_examples, target_length, sample_rate):
    organize_and_prepare_dataset(input_dir, output_dir, num_examples, target_length, sample_rate)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize and prepare stems into training dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing stem directories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed dataset.')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples to generate.')
    parser.add_argument('--target_length', type=int, default=60, help='Target length of each audio file in seconds.')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate for all audio files.')

    args = parser.parse_args()

    organize_and_prepare_dataset(args.input_dir, args.output_dir, args.num_examples, args.target_length, args.sample_rate)
