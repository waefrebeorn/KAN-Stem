import os
import torch
import torchaudio
import logging
from dataset import StemSeparationDataset  # Ensure this import matches your project structure
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

logger = logging.getLogger(__name__)

def preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, suppress_reading_messages, num_workers, device_prep, stop_flag):
    dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=target_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        apply_data_augmentation=apply_data_augmentation,
        suppress_warnings=suppress_warnings,
        suppress_reading_messages=suppress_reading_messages,
        num_workers=num_workers,
        device_prep=device_prep,
        stop_flag=stop_flag
    )
    dataset.load_all_stems()

def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device_prep):
    try:
        wave, sr = torchaudio.load(file_path)
        wave = wave.to(device_prep)

        mel = mel_spectrogram(wave)
        mel = torch.nn.functional.interpolate(mel, size=(mel.size(1), target_length), mode='linear')

        # Ensure the input tensor has the correct number of dimensions and channels
        mel = mel.unsqueeze(0)  # Add batch dimension
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # Add channel dimension if not present

        # Move mel back to CPU after processing on GPU
        mel = mel.to('cpu')
        return mel
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None
