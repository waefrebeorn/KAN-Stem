import os
import torch
import torchaudio
import tempfile
import soundfile as sf
from model import load_model

def perform_separation(checkpoint_path, file_path, n_mels=64, target_length=256, n_fft=1024, num_stems=7):
    """Performs stem separation on a given audio file using the loaded model."""

    # Load checkpoint and get in_features
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    in_channels = model_state_dict['conv1.depthwise.weight'].shape[1]
    out_channels = model_state_dict['conv1.pointwise.weight'].shape[0]

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.abs().max() == 0:
        raise ValueError("The provided audio file is silent and cannot be processed.")

    # Ensure the waveform has the expected number of channels
    if waveform.shape[0] < in_channels:
        waveform = waveform.repeat(in_channels // waveform.shape[0], 1)
    elif waveform.shape[0] > in_channels:
        waveform = waveform.mean(dim=0, keepdim=True).repeat(in_channels, 1)

    # Load the model
    model = load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Process the audio in chunks
    chunk_size = sample_rate * target_length // n_mels
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
    separated_audio = [[] for _ in range(num_stems)]

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

        spectrogram = spectrogram.unsqueeze(0).to(device)

        with torch.no_grad():
            separated_spectrograms = model(spectrogram)  # Shape [batch_size, num_stems, channels, height, width]
        
        for j in range(num_stems):
            separated_spectrogram = separated_spectrograms[0, j].cpu()
            inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
            griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)

            if separated_spectrogram.dim() == 1:
                separated_spectrogram = separated_spectrogram.unsqueeze(0)

            separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrogram))

            if separated_waveform.dim() == 1:
                separated_waveform = separated_waveform.unsqueeze(0)

            separated_audio[j].append(separated_waveform[:, :chunk_waveform.shape[1]])

    separated_audio = [torch.cat(chunks, dim=1) for chunks in separated_audio]

    output_paths = []
    for i, stem in enumerate(separated_audio):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_stem_{i+1}.wav") as tmpfile:
            sf.write(tmpfile.name, stem.numpy().T, sample_rate)
            output_paths.append(tmpfile.name)

    return output_paths
