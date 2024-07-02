import os
import random
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def normalize_length(waveform, sample_rate, target_length):
    target_samples = target_length * sample_rate
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    else:
        repeats = target_samples // len(waveform)
        remainder = target_samples % len(waveform)
        return np.concatenate((np.tile(waveform, repeats), waveform[:remainder]))

def ensure_mono(waveform):
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform

def save_as_ogg(waveform, sample_rate, file_path):
    temp_wav_path = file_path.replace('.ogg', '.wav')
    sf.write(temp_wav_path, waveform, sample_rate)
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(file_path, format='ogg')
    os.remove(temp_wav_path)

def combine_and_shuffle_stems(input_dir, output_dir, num_examples=100, target_length=60):
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
    targets = {}

    for i in range(num_examples):
        combined_waveform = None
        max_length = target_length * 44100  # Assuming a sample rate of 44100 Hz

        for stem_name, files in file_lists.items():
            if i < len(files):
                file_idx = i
            else:
                file_idx = i % len(files)
            
            file = files[file_idx]
            file_path = os.path.join(input_dir, stem_name, file)
            try:
                waveform, sample_rate = sf.read(file_path, dtype='float32')
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            waveform = ensure_mono(waveform)
            normalized_waveform = normalize_length(waveform, sample_rate, target_length)
            
            if stem_name not in targets:
                targets[stem_name] = []
            targets[stem_name].append(normalized_waveform)

            if combined_waveform is None:
                combined_waveform = np.zeros(max_length, dtype=np.float32)
            
            combined_waveform[:len(normalized_waveform)] += normalized_waveform
        
        combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))

        data.append(combined_waveform)

        print(f"Generated example {i+1}")

    data = np.array(data)

    for idx, example in enumerate(data):
        file_path = os.path.join(output_dir, f'example_{idx+1:04d}.ogg')
        save_as_ogg(example, sample_rate, file_path)

    for stem_name, target_examples in targets.items():
        for idx, example in enumerate(target_examples):
            file_path = os.path.join(output_dir, f'{stem_name}_{idx+1:04d}.ogg')
            save_as_ogg(example, sample_rate, file_path)

    print(f"Saved data and targets to {output_dir}")
    
def organize_and_prepare_dataset(input_dir, output_dir, num_examples):
    combine_and_shuffle_stems(input_dir, output_dir, num_examples)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

def organize_and_prepare_dataset_gradio(input_dir, output_dir, num_examples):
    organize_and_prepare_dataset(input_dir, output_dir, num_examples)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize and prepare stems into training dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing stem directories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed dataset.')  
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples to generate.')

    args = parser.parse_args()

    organize_and_prepare_dataset(args.input_dir, args.output_dir, args.num_examples)
