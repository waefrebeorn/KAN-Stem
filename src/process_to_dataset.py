import os
import random
import soundfile as sf
import numpy as np
from pydub import AudioSegment

def normalize_length(waveform, sample_rate, target_length):
    target_samples = target_length * sample_rate
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    else:
        repeats = target_samples // len(waveform)
        remainder = target_samples % len(waveform)
        return np.tile(waveform, repeats)[:target_samples]

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

def combine_and_shuffle_stems(input_dir, output_dir, num_examples=100, target_length=180):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.ogg', '.flac'))]
    print(f"Found {len(files)} audio files in {input_dir}")
    
    if len(files) < 4:
        raise ValueError("Not enough audio files to sample from. Need at least 4 files.")

    data = []
    targets = []

    for i in range(num_examples):
        selected_files = random.sample(files, 4)
        combined_waveform = None
        target_waveforms = []

        for file in selected_files:
            file_path = os.path.join(input_dir, file)
            waveform, sample_rate = sf.read(file_path, dtype='float32')
            waveform = ensure_mono(waveform)
            normalized_waveform = normalize_length(waveform, sample_rate, target_length)
            target_waveforms.append(normalized_waveform)

            if combined_waveform is None:
                combined_waveform = np.zeros_like(normalized_waveform)
            
            combined_waveform += normalized_waveform
        
        combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))

        data.append(combined_waveform)
        targets.append(target_waveforms)

        print(f"Generated example {i+1}")

    data = np.array(data)
    targets = np.array(targets)

    for idx, example in enumerate(data):
        file_path = os.path.join(output_dir, f'example_{idx+1}.ogg')
        save_as_ogg(example, sample_rate, file_path)

    for idx, example in enumerate(targets):
        for jdx, target in enumerate(example):
            file_path = os.path.join(output_dir, f'target_{idx+1}_{jdx+1}.ogg')
            save_as_ogg(target, sample_rate, file_path)

    print(f"Saved data and targets to {output_dir}")

def parse_and_organize_stems(input_dir, output_dir):
    keywords = {
        'vocals': ['vocals', 'vox', 'vocal'],
        'drums': ['kick', 'snare', 'cymbal', 'hihat', 'drums', 'drum'],
        'bass': ['bass', 'bs'],
        'guitar': ['guitar', 'gtr', 'gt'],
        'keys': ['keys', 'key', 'keyboard', 'synth'],
        'other': ['backing', 'bg', 'misc']
    }
    
    for key in keywords.keys():
        key_dir = os.path.join(output_dir, key)
        if not os.path.exists(key_dir):
            os.makedirs(key_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.ogg', '.flac'))]
    for file in files:
        found = False
        for key, words in keywords.items():
            if any(word in file.lower() for word in words):
                file_path = os.path.join(input_dir, file)
                target_path = os.path.join(output_dir, key, file)
                os.rename(file_path, target_path)
                found = True
                break
        if not found:
            file_path = os.path.join(input_dir, file)
            target_path = os.path.join(output_dir, 'other', file)
            os.rename(file_path, target_path)
    
    print("Stems organized by type.")

def organize_and_prepare_dataset(input_dir, output_dir, num_examples):
    parse_and_organize_stems(input_dir, output_dir)
    combine_and_shuffle_stems(output_dir, output_dir, num_examples)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

# Add a function to wrap the Gradio interface
def organize_and_prepare_dataset_gradio(input_dir, output_dir, num_examples):
    organize_and_prepare_dataset(input_dir, output_dir, num_examples)
    return f"Dataset prepared with {num_examples} examples in {output_dir}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize and prepare stems into training dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with unorganized stems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed dataset.')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples to generate.')

    args = parser.parse_args()

    organize_and_prepare_dataset(args.input_dir, args.output_dir, args.num_examples)
