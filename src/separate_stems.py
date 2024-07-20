import os
import torch
import torch.nn.functional as F
import logging
import soundfile as sf
import numpy as np
import librosa
from typing import List
from model import load_model
from utils import ensure_dir_exists, segment_audio, purge_vram
import torchaudio.transforms as T
from torch.utils.checkpoint import checkpoint

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="expandable_segments not supported on this platform")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")

logger = logging.getLogger(__name__)

def read_audio(file_path, suppress_messages=False):
    try:
        if not suppress_messages:
            logger.info(f"Attempting to read: {file_path}")
        data, samplerate = sf.read(file_path)
        return torch.tensor(data).unsqueeze(0), samplerate
    except FileNotFoundError:
        logger.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
    except sf.LibsndfileError as e:
        logger.error(f"Error decoding audio file {file_path}: {e}")
    return None, None

def write_audio(file_path, data, samplerate):
    try:
        data_cpu = data.squeeze(0).cpu().numpy()
        sf.write(file_path, data_cpu, samplerate)
    except Exception as e:
        logger.error(f"Error writing audio file {file_path}: {e}")

def process_segment(segment_tensor, sr, n_mels, n_fft, model, device):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=n_fft // 4
    ).to(device)(segment_tensor)

    harmonic, percussive = librosa.decompose.hpss(mel_spectrogram.squeeze().cpu().numpy())
    harmonic_tensor = torch.tensor(harmonic).unsqueeze(0).to(device)
    percussive_tensor = torch.tensor(percussive).unsqueeze(0).to(device)
    
    input_mel = torch.cat([mel_spectrogram, harmonic_tensor, percussive_tensor], dim=1)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            output_mel = checkpoint(model, input_mel, use_reentrant=False)

    # Ensure the types match for InverseMelScale
    output_mel = output_mel.float()

    inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels).to(device)
    griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32).to(device)

    audio = griffin_lim_transform(inverse_mel_transform(output_mel.squeeze(0))).cpu().numpy()
    
    purge_vram()
    return audio

def perform_separation(checkpoints, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages, batch_size, segment_length):
    logger.info("Loading model for separation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_audio, sr = read_audio(file_path, suppress_messages=suppress_reading_messages)
    if input_audio is None:
        logger.error(f"Error reading input audio from {file_path}")
        return []

    segment_length_samples = segment_length  # Use the provided segment length
    segments = segment_audio(input_audio.squeeze().numpy(), chunk_size=segment_length_samples)

    output_audio = []

    for checkpoint_path in checkpoints:
        model = load_model(checkpoint_path, 3, 3, n_mels, target_length, device=device)
        model.eval()
        
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}")
            segment_tensor = torch.tensor(segment).float().unsqueeze(0).to(device)

            if segment_tensor.size(-1) < segment_length_samples:
                segment_tensor = F.pad(segment_tensor, (0, segment_length_samples - segment_tensor.size(-1)))

            audio = process_segment(segment_tensor, sr, n_mels, n_fft, model, device)
            output_audio.append(audio)

            # Clear memory after processing each segment
            del segment_tensor
            del audio
            torch.cuda.empty_cache()
            purge_vram()

    result_paths = []
    ensure_dir_exists(cache_dir)
    
    # Combine segments and write to file
    full_audio = np.concatenate(output_audio, axis=-1)
    result_path = os.path.join(cache_dir, "separated_audio.wav")
    write_audio(result_path, torch.tensor(full_audio), sr)
    result_paths.append(result_path)

    return result_paths

if __name__ == '__main__':
    checkpoints = ['path_to_checkpoint_1', 'path_to_checkpoint_2']
    file_path = 'path_to_input_audio.wav'
    n_mels = 80
    target_length = 22050
    n_fft = 1024
    num_stems = 6
    cache_dir = './cache'
    suppress_reading_messages = False
    batch_size = 1  # Add this line
    segment_length = 22050  # Add this line
    
    separated_files = perform_separation(checkpoints, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages, batch_size, segment_length)
    logger.info(f'Separated files: {separated_files}')
