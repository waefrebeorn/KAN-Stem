import os
import random
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import warnings
import librosa

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def standardize_length(waveform, sample_rate, target_length):
    target_samples = target_length * sample_rate
    waveform_length = len(waveform)

    if waveform_length >= target_samples:
        return waveform[:target_samples]
    else:
        return np.pad(waveform, (0, target_samples - waveform_length), 'constant')

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

def generate_shuffled_noise(input_dirs, output_dir, num_examples=100, target_length=180):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stem_types = list(input_dirs.keys())
    files_dict = {stem: [os.path.join(input_dirs[stem], f) for f in os.listdir(input_dirs[stem]) if f.endswith(('.wav', '.ogg', '.flac'))] for stem in stem_types}
    
    print(f"Generating shuffled noise from stems.")

    for stem, files in files_dict.items():
        print(f"Found {len(files)} audio files in {stem} stem")

    data = []

    for i in range(num_examples):
        combined_waveform = None
        sample_rate = None

        for stem in stem_types:
            file = random.choice(files_dict[stem])
            file_path = os.path.join(input_dirs[stem], file)
            waveform, sample_rate = sf.read(file_path, dtype='float32')
            waveform = ensure_mono(waveform)
            normalized_waveform = standardize_length(waveform, sample_rate, target_length)
            
            shuffled_waveform = np.random.permutation(normalized_waveform)  # Shuffle the waveform
            
            if combined_waveform is None:
                combined_waveform = np.zeros_like(shuffled_waveform)
            
            combined_waveform += shuffled_waveform  # Combine shuffled noise

        # Normalize to prevent clipping
        combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))

        data.append(combined_waveform)
        
        print(f"Generated shuffled noise example {i+1}")

    data = np.array(data)

    for idx, example in enumerate(data):
        file_path = os.path.join(output_dir, f'shuffled_noise_{idx+1}.ogg')
        save_as_ogg(example, sample_rate, file_path)

    print(f"Saved shuffled noise data to {output_dir}")

# Add a function to wrap the Gradio interface
def generate_shuffled_noise_gradio(input_dir, output_dir, num_examples):
    input_dirs = {
        'vocals': os.path.join(input_dir, 'vocals'),
        'drums': os.path.join(input_dir, 'drums'),
        'bass': os.path.join(input_dir, 'bass'),
        'guitar': os.path.join(input_dir, 'guitar'),
        'keys': os.path.join(input_dir, 'keys'),
        'other': os.path.join(input_dir, 'other')
    }

    generate_shuffled_noise(input_dirs, output_dir, num_examples)
    return f"Generated {num_examples} shuffled noise examples in {output_dir}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate shuffled noise examples for 'other' category.")
    parser.add_argument('--input_dir', type=str, required=True, help='Base directory with organized stems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for shuffled noise dataset.')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of shuffled noise examples to generate.')

    args = parser.parse_args()

    input_dirs = {
        'vocals': os.path.join(args.input_dir, 'vocals'),
        'drums': os.path.join(args.input_dir, 'drums'),
        'bass': os.path.join(args.input_dir, 'bass'),
        'guitar': os.path.join(args.input_dir, 'guitar'),
        'keys': os.path.join(args.input_dir, 'keys'),
        'other': os.path.join(args.input_dir, 'other')
    }

    generate_shuffled_noise(input_dirs, args.output_dir, args.num_examples)

