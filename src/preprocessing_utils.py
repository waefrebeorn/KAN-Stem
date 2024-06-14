import os
import torch
import torchaudio
import logging
from torchaudio import transforms as T
import h5py
import numpy as np
import librosa

logger = logging.getLogger(__name__)

def calculate_harmonic_content(spectrogram):
    spectrogram_np = spectrogram.cpu().detach().numpy()
    harmonic, _ = librosa.decompose.hpss(spectrogram_np)
    return torch.from_numpy(harmonic)

def calculate_percussive_content(spectrogram):
    spectrogram_np = spectrogram.cpu().detach().numpy()
    _, percussive = librosa.decompose.hpss(spectrogram_np)
    return torch.from_numpy(percussive)

def preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, suppress_reading_messages, num_workers, device, stop_flag):
    os.makedirs(cache_dir, exist_ok=True)
    
    valid_stems = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    for file_name in valid_stems:
        if stop_flag.value == 1:
            return

        file_path = os.path.join(data_dir, file_name)
        cache_key = _get_cache_key(file_path, target_length, apply_data_augmentation)
        cache_file_path = os.path.join(cache_dir, f"{cache_key}.h5")

        if os.path.exists(cache_file_path):
            try:
                data = load_from_cache(cache_file_path)
                if not _validate_data(data):
                    os.remove(cache_file_path)
                    raise ValueError(f"Invalid data in cache for {file_name}, cache will be recreated.")
            except Exception as e:
                logger.warning(f"Cache loading failed for {file_name}, recreating cache. Error: {e}")
                os.remove(cache_file_path)

        try:
            mel_spectrogram = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=n_fft,
                win_length=None,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                power=2.0
            ).to(torch.float32).to(device)

            input_mel, target_mel = load_and_preprocess(
                file_path, mel_spectrogram, target_length, apply_data_augmentation, device,
                cache_dir=cache_dir, use_cache=False,
                extra_features=[calculate_harmonic_content, calculate_percussive_content]
            )

            if input_mel is not None:
                input_mel = input_mel.to(torch.float32)
                target_mel = input_mel.clone()

                with h5py.File(cache_file_path, 'w') as f:
                    f.create_dataset('input', data=input_mel.cpu().detach().numpy())
                    f.create_dataset('target', data=target_mel.cpu().detach().numpy())

                logger.info(f"Cached: {cache_file_path}")

        except Exception as e:
            if not suppress_warnings:
                logger.warning(f"Skipping {file_name} due to error: {e}")

def load_from_cache(cache_file_path):
    try:
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.from_numpy(f['input'][:])
            target_data = torch.from_numpy(f['target'][:])
        return input_data, target_data
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        raise

def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device, cache_dir='./cache', use_cache=True, extra_features=None):
    try:
        logger.debug("Starting load and preprocess")

        if use_cache:
            cache_key = _get_cache_key(file_path, target_length, apply_data_augmentation)
            cache_file_path = os.path.join(cache_dir, f"{cache_key}.h5")

            if os.path.exists(cache_file_path):
                try:
                    return load_from_cache(cache_file_path)
                except Exception as e:
                    logger.error(f"Error loading from cache, recreating. Error: {e}")
                    os.remove(cache_file_path)

        input_audio, _ = torchaudio.load(file_path)
        if input_audio is None:
            raise ValueError("Failed to load audio")

        logger.debug(f"Loaded audio with shape: {input_audio.shape}")

        if apply_data_augmentation:
            input_audio = data_augmentation(input_audio.to(device))
            logger.debug(f"After data augmentation, audio shape: {input_audio.shape}")

        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]
        target_mel = input_mel.clone()

        logger.debug(f"Input mel spectrogram shape: {input_mel.shape}")

        extra_feature_tensors = []
        if extra_features:
            for feature_fn in extra_features:
                feature_tensor = feature_fn(input_mel)
                extra_feature_tensors.append(feature_tensor)

        input_mel = torch.stack([input_mel] + extra_feature_tensors, dim=-1)

        logger.debug(f"4D input spectrogram shape: {input_mel.shape}")

        input_mel = input_mel.to('cpu')
        target_mel = target_mel.to('cpu')

        if input_mel.shape[1] < target_length:
            padding = target_length - input_mel.shape[1]
            input_mel = torch.nn.functional.pad(input_mel, (0, padding, 0, 0), mode='constant')
            target_mel = torch.nn.functional.pad(target_mel, (0, padding), mode='constant')

        logger.debug(f"Padded input mel shape: {input_mel.shape}")
        logger.debug(f"Padded target mel shape: {target_mel.shape}")

        if use_cache:
            _save_to_hdf5_cache(cache_file_path, input_mel, target_mel)

        return input_mel, target_mel

    except Exception as e:
        logger.error(f"Error in load and preprocess: {e}")
        return None, None

def data_augmentation(inputs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        pitch_shift = T.PitchShift(sample_rate=22050, n_steps=2).to(device)
        freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
        time_mask = T.TimeMasking(time_mask_param=35).to(device)

        augmented_inputs = pitch_shift(inputs)
        augmented_inputs = freq_mask(augmented_inputs)
        augmented_inputs = time_mask(augmented_inputs)

        return augmented_inputs
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        return inputs

def _get_cache_key(file_path, target_length, apply_data_augmentation):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return f"{file_name}_{target_length}_{apply_data_augmentation}"

def _save_to_hdf5_cache(cache_file_path, input_data, target_data):
    try:
        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
        with h5py.File(cache_file_path, 'w') as f:
            f.create_dataset('input', data=input_data.detach().cpu().numpy())
            f.create_dataset('target', data=target_data.detach().cpu().numpy())
        logger.info(f"Saved cache: {cache_file_path}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def _validate_data(data):
    try:
        if not isinstance(data, tuple) or len(data) != 2:
            return False

        input_data, target_data = data
        if not isinstance(input_data, torch.Tensor) or not isinstance(target_data, torch.Tensor):
            return False

        input_shape = input_data.shape
        target_shape = target_data.shape
        if input_shape[:-1] != target_shape:
            return False

        return True
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        return False
