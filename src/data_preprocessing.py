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

def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device, cache_dir, use_cache=True):
    try:
        logger.debug("Starting load and preprocess")
        
        if use_cache:
            cache_path = os.path.join(cache_dir, f"{os.path.basename(file_path)}.h5")
            if os.path.exists(cache_path):
                logger.info(f"Loading from cache: {cache_path}")
                return load_from_cache(cache_path)
            else:
                logger.info(f"Cache not found for {file_path}, processing on the fly")

        input_audio, _ = torchaudio.load(file_path)
        if input_audio is None:
            return None

        logger.debug(f"Device for processing: {device}")

        if apply_data_augmentation:
            input_audio = input_audio.float().to(device)
            logger.debug(f"input_audio device: {input_audio.device}")
            input_audio = data_augmentation(input_audio, device=device)
            logger.debug(f"After data augmentation, input_audio device: {input_audio.device}")
        else:
            input_audio = input_audio.float().to(device)

        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]

        # Move input_mel back to CPU after processing on GPU
        input_mel = input_mel.to('cpu')
        logger.debug(f"input_mel device: {input_mel.device}")

        logger.debug("Completed load and preprocess")

        # Save to cache if caching is enabled
        if use_cache:
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('mel_spectrogram', data=input_mel.numpy())
            logger.info(f"Cached: {cache_path}")

        return input_mel

    except Exception as e:
        logger.error(f"Error in load and preprocess: {e}")
        return None

def data_augmentation(inputs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        pitch_shift = torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=2).to(device)
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15).to(device)
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35).to(device)

        augmented_inputs = pitch_shift(inputs.clone().detach().to(device))
        augmented_inputs = freq_mask(augmented_inputs.clone().detach())
        augmented_inputs = time_mask(augmented_inputs.clone().detach())

        return augmented_inputs
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        return inputs
