import os
import torch
import torchaudio
import tempfile
import soundfile as sf
import torch.nn.functional as F
import torch.nn as nn
from model import load_model

def perform_separation(checkpoint_path, file_path, n_mels=64, target_length=256, n_fft=1024, num_stems=7, cache_dir='./cache'):
    """Performs stem separation on a given audio file using the loaded model."""

    stems = ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]
    separated_audio = {stem: [] for stem in stems}

    # Load and preprocess audio
    print("Loading audio...")
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.abs().max() == 0:
        raise ValueError("The provided audio file is silent and cannot be processed.")

    # Convert to mono if necessary
    if waveform.size(0) > 1:  # If stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("Converted stereo to mono.")

    # Load the model
    print("Loading model...")
    model = load_model(checkpoint_path, 1, 64, n_mels, target_length, num_stems, 'cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate chunk size based on the model's input requirements
    chunk_size = sample_rate * target_length // n_mels
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
    print(f"Processing audio in {num_chunks} chunks of size {chunk_size}...")

    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}...")
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, waveform.shape[1])
        chunk_waveform = waveform[:, start_idx:end_idx]

        if chunk_waveform.shape[1] < chunk_size:
            padding = chunk_size - chunk_waveform.shape[1]
            chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding))
            print(f"Padded chunk {i+1} with {padding} samples.")

        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft
        )(chunk_waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        spectrogram = spectrogram.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension

        # Perform forward pass with dynamic adjustment of fc1
        with torch.no_grad():
            x = spectrogram
            x = model.pool1(F.relu(model.conv1(x)))
            x = model.pool2(F.relu(model.conv2(x)))
            x = model.pool3(F.relu(model.conv3(x)))
            x = model.pool4(F.relu(model.conv4(x)))
            x = model.pool5(x)
            x = model.flatten(x)
            actual_conv_output_size = x.shape[1]

            if model.fc1 is None or model.fc1.in_features != actual_conv_output_size:
                model.fc1 = nn.Linear(actual_conv_output_size, 1024).to(spectrogram.device)
                model.fc2 = nn.Linear(1024, model.n_mels * model.target_length * model.num_stems).to(spectrogram.device)
                print(f"Adjusted fc1 and fc2 for chunk {i+1}.")

            # Now perform the forward pass
            separated_spectrograms = model(spectrogram).squeeze(0).cpu()

        inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
        griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)

        for j, stem in enumerate(stems):
            separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrograms[j]))
            separated_audio[stem].append(separated_waveform[:chunk_waveform.shape[1]])
            print(f"Processed stem {stem} for chunk {i+1}.")

    output_paths = []
    for stem, audio_chunks in separated_audio.items():
        final_waveform = torch.cat(audio_chunks, dim=1 if len(audio_chunks[0].shape) > 1 else 0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.wav") as tmpfile:
            sf.write(tmpfile.name, final_waveform.numpy().T, sample_rate)
            output_paths.append(tmpfile.name)
            print(f"Saved {stem} output to {tmpfile.name}")

    print("Separation completed.")
    return output_paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Perform stem separation on an audio file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--file', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bands')
    parser.add_argument('--target_length', type=int, default=256, help='Target length of the output')
    parser.add_argument('--n_fft', type=int, default=1024, help='Number of FFT components')
    parser.add_argument('--num_stems', type=int, default=7, help='Number of stems to separate')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory')

    args = parser.parse_args()

    perform_separation(args.checkpoint, args.file, args.n_mels, args.target_length, args.n_fft, args.num_stems, args.cache_dir)
