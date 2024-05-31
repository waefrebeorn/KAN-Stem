import os
import random
import soundfile as sf
import numpy as np

def normalize_length(waveform, sample_rate, target_length):
    """Normalize the audio length to target_length seconds by looping if necessary."""
    target_samples = int(target_length * sample_rate)
    current_samples = waveform.shape[0]
    
    if current_samples > target_samples:
        return waveform[:target_samples]
    else:
        repeats = target_samples // current_samples
        remainder = target_samples % current_samples
        return np.concatenate([np.tile(waveform, repeats), waveform[:remainder]])

def ensure_mono(waveform):
    """Ensure the waveform is mono by averaging across channels if necessary."""
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform

def combine_stems(input_dirs, output_dir, num_examples=100, target_length=180):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stem_types = list(input_dirs.keys())
    files_dict = {stem: [os.path.join(input_dirs[stem], f) for f in os.listdir(input_dirs[stem]) if f.endswith(('.wav', '.flac'))] for stem in stem_types}
    min_file_count = min(len(files) for files in files_dict.values())
    
    print(f"Found minimum {min_file_count} audio files in each stem type")

    data = []
    targets = []

    for i in range(num_examples):
        combined_waveform = None
        sample_rate = None
        target_waveforms = []

        for stem in stem_types:
            file = random.choice(files_dict[stem])
            file_path = os.path.join(input_dirs[stem], file)
            waveform, sample_rate = sf.read(file_path, dtype='float32')
            waveform = ensure_mono(waveform)
            normalized_waveform = normalize_length(waveform, sample_rate, target_length)
            
            target_waveforms.append(normalized_waveform)
            
            if combined_waveform is None:
                combined_waveform = np.zeros_like(normalized_waveform)
            
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process organized stems into training dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Base directory with organized stems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed dataset.')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples to generate.')

    args = parser.parse_args()

    input_dirs = {
        'vocals': os.path.join(args.input_dir, 'vocals'),
        'drums': os.path.join(args.input_dir, 'drums'),
        'bass': os.path.join(args.input_dir, 'bass'),
        'guitar': os.path.join(args.input_dir, 'guitar'),
        'keys': os.path.join(args.input_dir, 'keys'),
        'other': os.path.join(args.input_dir, 'other')
    }

    result_message = combine_stems(input_dirs, args.output_dir, args.num_examples)
    print(result_message)
