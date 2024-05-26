import os
import torch
import torchaudio
import tempfile
import soundfile as sf
from model import load_model

def perform_separation(checkpoint_dir, file_path, n_mels=64, target_length=256, n_fft=1024, num_stems=7, cache_dir='./cache'):
    """Performs stem separation on a given audio file using the loaded model."""
    
    stems = ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]
    separated_audio = {stem: [] for stem in stems}
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.abs().max() == 0:
        raise ValueError("The provided audio file is silent and cannot be processed.")

    chunk_size = sample_rate * target_length // n_mels
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size

    for stem in stems:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{stem}.pt')
        model = load_model(checkpoint_path, 1, 64, n_mels, target_length)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, waveform.shape[1])
            chunk_waveform = waveform[:, start_idx:end_idx]

            if chunk_waveform.shape[1] < chunk_size:
                padding = chunk_size - chunk_waveform.shape[1]
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding))

            spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft
            )(chunk_waveform)
            spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

            spectrogram = spectrogram.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                separated_spectrogram = model(spectrogram).squeeze(0).cpu()

            inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
            griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)

            separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrogram))
            separated_audio[stem].append(separated_waveform[:, :chunk_waveform.shape[1]])

    output_paths = []
    for stem, audio_chunks in separated_audio.items():
        final_waveform = torch.cat(audio_chunks, dim=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.wav") as tmpfile:
            sf.write(tmpfile.name, final_waveform.numpy().T, sample_rate)
            output_paths.append(tmpfile.name)

    return output_paths
