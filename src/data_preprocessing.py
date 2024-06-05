import os
import torch
import torchaudio
import logging
from dataset import StemSeparationDataset  # Ensure this import matches your project structure
from preprocessing_utils import load_and_preprocess  # Import from the new location
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
