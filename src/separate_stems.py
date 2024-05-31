import os
import torch
import torchaudio.transforms as T
import logging
from model import load_model

logger = logging.getLogger(__name__)

def perform_separation(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    logger.info("Loading model for separation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, 1, 64, n_mels, target_length, num_stems, device)
    model.eval()

    input_audio, sr = read_audio(file_path, suppress_messages=suppress_reading_messages)
    if input_audio is None:
        logger.error(f"Error reading input audio from {file_path}")
        return []

    input_mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(input_audio.float()).unsqueeze(0).to(device)
    output_mel = model(input_mel).cpu()

    inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32)
    
    output_audio = []
    for i in range(num_stems):
        mel = output_mel[:, i, :, :]
        audio = griffin_lim_transform(inverse_mel_transform(mel)).numpy()
        output_audio.append(audio)

    result_paths = []
    for i, audio in enumerate(output_audio):
        result_path = os.path.join(cache_dir, f"separated_stem_{i}.wav")
        write_audio(result_path, torch.tensor(audio), sr)
        result_paths.append(result_path)

    return result_paths
