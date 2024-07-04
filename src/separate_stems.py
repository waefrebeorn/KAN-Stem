import os
import torch
import torch.nn as nn
import torchaudio.transforms as T
import logging
import soundfile as sf
import librosa
from model import load_model
from utils import purge_vram  # Only importing purge_vram from utils

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

def perform_separation(checkpoints, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    logger.info("Loading model for separation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_audio, sr = read_audio(file_path, suppress_messages=suppress_reading_messages)
    if input_audio is None:
        logger.error(f"Error reading input audio from {file_path}")
        return []

    output_audio = []

    for checkpoint_path in checkpoints:
        model = load_model(checkpoint_path, 3, 3, n_mels, target_length, device=device)
        model.eval()
        
        with torch.no_grad():
            mel_spectrogram = T.MelSpectrogram(
                sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=n_fft // 4
            )(input_audio.float()).unsqueeze(0).to(device)
            
            # Calculate harmonic and percussive content and stack into 4D spectrogram
            spectrogram_np = mel_spectrogram.cpu().detach().numpy()
            harmonic, percussive = librosa.decompose.hpss(spectrogram_np[0])
            harmonic_t = torch.from_numpy(harmonic).unsqueeze(0).to(device)
            percussive_t = torch.from_numpy(percussive).unsqueeze(0).to(device)
            
            input_mel = torch.stack([mel_spectrogram, harmonic_t, percussive_t], dim=-1)
            
            output_mel = model(input_mel).cpu()
            inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
            griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32)
            audio = griffin_lim_transform(inverse_mel_transform(output_mel.squeeze(0))).numpy()
            output_audio.append(audio)

    result_paths = []
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    for i, audio in enumerate(output_audio):
        result_path = os.path.join(cache_dir, f"separated_stem_{i}.wav")
        write_audio(result_path, torch.tensor(audio), sr)
        result_paths.append(result_path)

    return result_paths

if __name__ == '__main__':
    checkpoints = ['path_to_checkpoint_1', 'path_to_checkpoint_2']
    file_path = 'path_to_input_audio.wav'
    n_mels = 128
    target_length = 256
    n_fft = 2048
    num_stems = 6
    cache_dir = './cache'
    suppress_reading_messages = False
    
    separated_files = perform_separation(checkpoints, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages)
    logger.info(f'Separated files: {separated_files}')
