import os
import h5py
import torch
import torchaudio
import logging
from multiprocessing import Lock

logger = logging.getLogger(__name__)

def preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, suppress_reading_messages, num_workers, device, stop_flag):
    os.makedirs(cache_dir, exist_ok=True)
    
    # Preprocess and cache data
    for file_name in os.listdir(data_dir):
        if stop_flag.value == 1:
            return

        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            cache_path = os.path.join(cache_dir, f"{file_name}.h5")

            if os.path.exists(cache_path):
                continue

            try:
                data, sample_rate = torchaudio.load(file_path)
                data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=n_fft)(data)
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft)(data)

                if apply_data_augmentation:
                    # Apply data augmentation if needed
                    pass

                # Save to HDF5 file
                with h5py.File(cache_path, 'w') as f:
                    f.create_dataset('mel_spectrogram', data=mel_spectrogram.numpy())

                logger.info(f"Cached: {cache_path}")

            except Exception as e:
                if not suppress_warnings:
                    logger.warning(f"Skipping {file_name} due to error: {e}")

def load_from_cache(cache_path):
    with h5py.File(cache_path, 'r') as f:
        mel_spectrogram = torch.tensor(f['mel_spectrogram'][:])
    return mel_spectrogram
