import os
import random
import soundfile as sf
import numpy as np

def normalize_length(waveform, sample_rate, target_length):
    """Normalize the audio length to target_length seconds by looping if necessary."""
    target_samples = target_length * sample_rate
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    else:
        repeats = target_samples // len(waveform)
        remainder = target_samples % len(waveform)
        return np.tile(waveform, repeats)[:target_samples]

def combine_and_shuffle_stems(input_dir, output_dir, num_examples=100, target_length=180):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.flac'))]
    print(f"Found {len(files)} audio files in {input_dir}")
    
    if len(files) < 4:
        raise ValueError("Not enough audio files to sample from. Need at least 4 files.")

    data = []
    targets = []

    for i in range(num_examples):
        selected_files = random.sample(files, 4)  # Select 4 random stems
        combined_waveform = None
        target_waveforms = []

        for file in selected_files:
            file_path = os.path.join(input_dir, file)
            waveform, sample_rate = sf.read(file_path, dtype='float32')
            normalized_waveform = normalize_length(waveform, sample_rate, target_length)
            target_waveforms.append(normalized_waveform)

            if combined_waveform is None:
                combined_waveform = normalized_waveform
            else:
                combined_waveform += normalized_waveform  # Simple sum to combine audio data
        
        # Normalize to prevent clipping
        combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))

        data.append(combined_waveform)
        targets.append(target_waveforms)
        
        print(f"Generated example {i+1}")

    data = np.array(data)
    targets = np.array(targets)

    np.save(os.path.join(output_dir, 'data.npy'), data)
    np.save(os.path.join(output_dir, 'targets.npy'), targets)

    print(f"Saved data and targets to {output_dir}")

def parse_and_organize_stems(input_dir, output_dir):
    keywords = {
        'vocals': ['vocals', 'vox', 'vocal'],
        'drums': ['drums', 'drum', 'dr'],
        'bass': ['bass', 'bs'],
        'guitar': ['guitar', 'gtr', 'gt'],
        'keys': ['keys', 'key', 'keyboard', 'synth'],
        'other': []
    }
    
    for key in keywords.keys():
        key_dir = os.path.join(output_dir, key)
        if not os.path.exists(key_dir):
            os.makedirs(key_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.flac'))]
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
    return f"Dataset prepared with {num_examples} examples in {output_dir}."

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize and prepare dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with audio files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for organized and prepared dataset.')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples to generate.')

    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    result_message = organize_and_prepare_dataset(args.input_dir, args.output_dir, args.num_examples)
    print(result_message)
