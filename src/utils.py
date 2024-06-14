import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import soundfile as sf
import psutil
import GPUtil
import time
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as T
import h5py
import numpy as np
import librosa
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

logger = logging.getLogger(__name__)

def setup_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def detect_parameters(data_dir, default_n_mels=64, default_n_fft=1024):
    sample_rates = []
    durations = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            sample_rate, duration, is_silent = analyze_audio(file_path)
            if is_silent:
                logger.info(f"Skipping silent file: {file_name}")
                continue
            sample_rates.append(sample_rate)
            durations.append(duration)

    logger.info(f"Found {len(sample_rates)} valid audio files")

    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    avg_sample_rate = sum(sample_rates) / len(sample_rates)
    avg_duration = sum(durations) / len(durations)

    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))

    return int(avg_sample_rate), n_mels, n_fft

def analyze_audio(file_path):
    try:
        data, sample_rate = sf.read(file_path)
        duration = len(data) / sample_rate
        is_silent = (abs(data).max() < 1e-5)
        return sample_rate, duration, is_silent
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        return None, None, True

def convert_to_3_channels(tensor):
    """Convert a single-channel tensor to a 3-channel tensor by repeating the single channel."""
    return tensor.repeat(1, 3, 1, 1)

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, n_ffts=[2048, 1024, 512, 256, 128]):
        super(MultiScaleSpectralLoss, self).__init__()
        self.n_ffts = n_ffts

    def forward(self, y_true, y_pred):
        loss = 0.0
        for n_fft in self.n_ffts:
            y_true_stft = torch.stft(y_true, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, return_complex=True)
            y_pred_stft = torch.stft(y_pred, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, return_complex=True)
            loss += F.l1_loss(torch.abs(y_true_stft), torch.abs(y_pred_stft))
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * len(self.feature_extractor)

    def forward(self, y_true, y_pred):
        loss = 0.0
        features_true = self.feature_extractor(y_true)
        features_pred = self.feature_extractor(y_pred)
        for weight, feature_true, feature_pred in zip(self.layer_weights, features_true, features_pred):
            loss += weight * F.l1_loss(feature_true, feature_pred)
        return loss

def compute_adversarial_loss(discriminator, y_true, y_pred, device):
    y_true = y_true.unsqueeze(1) if y_true.dim() == 3 else y_true
    y_pred = y_pred.unsqueeze(1) if y_pred.dim() == 3 else y_pred
    real_labels = torch.ones(y_true.size(0), 1, device=device)
    fake_labels = torch.zeros(y_pred.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy_with_logits(discriminator(y_true), real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(y_pred), fake_labels)
    return real_loss, fake_loss

def gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = alpha * real_data + ((1 - alpha) * fake_data)
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def log_system_resources():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None

        logger.info(f"CPU usage: {cpu_percent}%")
        logger.info(f"Memory usage: {memory_info.percent}%")
        if gpu_info:
            logger.info(f"GPU usage: {gpu_info.load * 100:.2f}%")
            logger.info(f"GPU memory usage: {gpu_info.memoryUtil * 100:.2f}%")
    except Exception as e:
        logger.error(f"Error logging system resources: {e}")

def process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device, queue):
    try:
        logger.info(f"Processing file: {file_path}")
        log_system_resources()
        start_time = time.time()

        result = load_and_preprocess(file_path, mel_spectrogram, target_length, device, apply_data_augmentation=apply_data_augmentation)
        if result is not None:
            pass

        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")

        queue.put(file_path)

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")

def worker_loop(file_paths, mel_spectrogram_params, target_length, apply_data_augmentation, queue):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params).to(device).float()

    try:
        for file_path in file_paths:
            logger.debug(f"Starting processing for file: {file_path}")
            process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device, queue)
            logger.debug(f"Finished processing for file: {file_path}")
        logger.debug("All files processed in worker loop")

        logger.debug("Entering post-processing stage after worker loop")
        logger.debug("Completed post-processing stage")

    except Exception as e:
        logger.error(f"Error in worker loop: {e}")

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for audio prediction."""
    try:
        mse = F.mse_loss(y_pred, y_true).item()
        snr = 10 * torch.log10(torch.mean(y_true ** 2) / torch.mean((y_true - y_pred) ** 2)).item()
        return {'mse': mse, 'snr': snr}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        raise e

def get_checkpoints(checkpoints_dir):
    """Retrieve all checkpoint files in the specified directory."""
    try:
        checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints
    except Exception as e:
        logger.error(f"Error retrieving checkpoints: {e}")
        return []

def compute_sdr(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    sdr = 10 * torch.log10(s_true / (s_noise + 1e-8))
    return sdr

def compute_sir(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_interf = torch.sum((true - noise) ** 2, dim=[1, 2, 3])
    sir = 10 * torch.log10(s_true / (s_interf + 1e-8))
    return sir

def compute_sar(true, pred):
    noise = true - pred
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    s_artif = torch.sum((pred - noise) ** 2, dim=[1, 2, 3])
    sar = 10 * torch.log10(s_noise / (s_artif + 1e-8))
    return sar

def log_training_parameters(params):
    """Logs the training parameters."""
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    if optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Momentum":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSProp":
        return optim.RMSprop(model_parameters, lr=learning_rate, alpha=0.99, weight_decay=weight_decay)
    elif optimizer_name == "Adadelta":
        return optim.Adadelta(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def wasserstein_loss(real_output, fake_output):
    return torch.mean(fake_output) - torch.mean(real_output)

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, suppress_reading_messages, num_workers, device_prep, stop_flag, use_cache=True):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_warnings = suppress_warnings
        self.suppress_reading_messages = suppress_reading_messages
        self.num_workers = num_workers
        self.device_prep = device_prep
        self.stop_flag = stop_flag
        self.use_cache = use_cache

        self.valid_stems = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        if not self.valid_stems:
            raise ValueError(f"No valid audio files found in {data_dir}")

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index = self._load_cache_metadata()

    def _load_cache_metadata(self):
        index_file_path = os.path.join(self.cache_dir, "cache_index.h5")
        cache_index = defaultdict(list)
        if os.path.exists(index_file_path):
            try:
                with h5py.File(index_file_path, 'r') as f:
                    for key in f.attrs:
                        cache_index[key] = list(f.attrs[key])
                if not self.suppress_reading_messages:
                    logger.info(f"Loaded cache metadata from {index_file_path}")
            except Exception as e:
                if not self.suppress_reading_messages:
                    logger.error(f"Error loading cache metadata file: {e}")
        return cache_index

    def _save_cache_metadata(self):
        index_file_path = os.path.join(self.cache_dir, "cache_index.h5")
        try:
            if not self.suppress_reading_messages:
                logger.info(f"Saving cache metadata to {index_file_path}")
            with h5py.File(index_file_path, 'w') as f:
                for key, val in self.cache_index.items():
                    f.attrs[key] = val
            if not self.suppress_reading_messages:
                logger.info(f"Successfully saved cache metadata to {index_file_path}")
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error saving cache metadata file: {e}")

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        if self.stop_flag.value == 1:
            return None

        stem_name = self.valid_stems[idx]
        try:
            data = self._get_data(stem_name, apply_data_augmentation=False)
            if data is None:
                data = self._process_and_cache(stem_name, apply_data_augmentation=False)
                if data is None:
                    raise ValueError(f"Failed to process and cache data for {stem_name}")

            if self.apply_data_augmentation:
                augmented_data = self._get_data(stem_name, apply_data_augmentation=True)
                if augmented_data is None:
                    augmented_data = self._process_and_cache(stem_name, apply_data_augmentation=True)
                    if augmented_data is None:
                        raise ValueError(f"Failed to process and cache augmented data for {stem_name}")
                logger.debug(f"Augmented input shape: {augmented_data['input'].shape}")
                logger.debug(f"Augmented target shape: {augmented_data['target'].shape}")
                return augmented_data

            logger.debug(f"Original input shape: {data['input'].shape}")
            logger.debug(f"Original target shape: {data['target'].shape}")
            return data

        except Exception as e:
            logger.error(f"Error in __getitem__ for stem {stem_name}: {e}")
            return None

    def _get_cache_key(self, stem_name, apply_data_augmentation):
        return f"{stem_name}_{apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"

    def _get_data(self, stem_name, apply_data_augmentation):
        cache_key = self._get_cache_key(stem_name, apply_data_augmentation)
        if self.use_cache and cache_key in self.cache_index:
            stem_cache_path = self.cache_index[cache_key]
            if isinstance(stem_cache_path, list):  # Check if stem_cache_path is a list
                stem_cache_path = stem_cache_path[0]  # Extract the first (and only) element

            if os.path.exists(stem_cache_path):
                try:
                    data = load_from_cache(stem_cache_path)
                    if self._validate_data(data):
                        if not self.suppress_reading_messages:
                            logger.info(f"Loaded cached data for {stem_name} (augmented={apply_data_augmentation})")
                        data['input'] = data['input'].to(self.device_prep).float()
                        data['target'] = data['target'].to(self.device_prep).float()

                        # Clear memory after loading cached data
                        torch.cuda.empty_cache()  # Clear GPU memory
                        import gc
                        gc.collect()  # Clear CPU memory

                        return data
                    else:
                        if not self.suppress_reading_messages:
                            logger.warning(f"Invalid cached data for {stem_name} (augmented={apply_data_augmentation}). Reprocessing.")
                except Exception as e:
                    if not self.suppress_reading_messages:
                        logger.warning(f"Error loading cached data for {stem_name} (augmented={apply_data_augmentation}). Reprocessing. Error: {e}")

        return None

    def _process_and_cache(self, stem_name, apply_data_augmentation):
        if self.stop_flag.value == 1:
            return None

        try:
            file_path = os.path.join(self.data_dir, stem_name)
            mel_spectrogram = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=self.n_fft,
                win_length=None,
                hop_length=self.n_fft // 4,
                n_mels=self.n_mels,
                power=2.0
            ).to(torch.float32).to(self.device_prep)

            extra_features = [calculate_harmonic_content, calculate_percussive_content]
            input_mel, target_mel = load_and_preprocess(file_path, mel_spectrogram, self.target_length, self.device_prep, cache_dir=self.cache_dir, use_cache=self.use_cache, extra_features=extra_features)
            if input_mel is None:
                raise ValueError(f"Failed to load and preprocess data for {stem_name}")

            input_mel = input_mel.to(torch.float32)
            target_mel = target_mel.to(torch.float32)
                
            if input_mel.shape[-1] < self.target_length:
                input_mel = torch.nn.functional.pad(input_mel, (0, self.target_length - input_mel.shape[-1]), mode='constant')

            if input_mel.shape[0] != target_mel.shape[0]:
                input_mel = input_mel.unsqueeze(0)  # Adding batch dimension if missing
            if target_mel.shape[0] != input_mel.shape[0]:
                target_mel = target_mel.unsqueeze(0)  # Adding batch dimension if missing

            data = {"input": input_mel, "target": target_mel}

            logger.debug(f"Processed input shape: {input_mel.shape}")
            logger.debug(f"Processed target shape: {target_mel.shape}")

            self._save_individual_cache(stem_name, data, apply_data_augmentation)

            return data

        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error processing stem {stem_name}: {e}")
            return None

    def _save_individual_cache(self, stem_name, data, apply_data_augmentation):
        cache_key = self._get_cache_key(stem_name, apply_data_augmentation)
        stem_cache_path = os.path.join(self.cache_dir, f"{cache_key}.h5")
        try:
            if not self.suppress_reading_messages:
                logger.info(f"Saving stem cache: {stem_cache_path}")
            with h5py.File(stem_cache_path, 'w') as f:
                f.create_dataset('input', data=data['input'].cpu().detach().numpy())
                f.create_dataset('target', data=data['target'].cpu().detach().numpy())
            if not self.suppress_reading_messages:
                logger.info(f"Successfully saved stem cache: {stem_cache_path}")
            self.cache_index[cache_key] = stem_cache_path
            self._save_cache_metadata()
            return stem_cache_path
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error saving stem cache: {stem_cache_path}. Error: {e}")
            return None

    def _validate_data(self, data):
        if 'input' not in data or 'target' not in data:
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: 'input' or 'target' key missing")
            return False

        if not isinstance(data['input'], torch.Tensor) or not isinstance(data['target'], torch.Tensor):
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: 'input' or 'target' is not a torch.Tensor. Input type: {type(data['input'])}, Target type: {type(data['target'])}")
            return False

        input_shape = data['input'].shape
        target_shape = data['target'].shape
        expected_shape = (1, self.n_mels, self.target_length)

        if input_shape != expected_shape or target_shape != expected_shape:
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: shape mismatch. Input shape: {input_shape}, Target shape: {target_shape}, Expected shape: {expected_shape}")
            return False

        return True

    def load_all_stems(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._load_stem_data, stem_name) for stem_name in self.valid_stems]
            for future in as_completed(futures):
                if self.stop_flag.value == 1:
                    break
                future.result()

    def _load_stem_data(self, stem_name):
        if self.stop_flag.value == 1:
            return None

        data = self._get_data(stem_name, apply_data_augmentation=False)
        if data is None:
            data = self._process_and_cache(stem_name, apply_data_augmentation=False)

        if self.apply_data_augmentation:
            augmented_data = self._get_data(stem_name, apply_data_augmentation=True)
            if augmented_data is None:
                augmented_data = self._process_and_cache(stem_name, apply_data_augmentation=True)
            return augmented_data

        return data

def pad_tensor(tensor, target_length, target_width):
    if tensor.dim() == 2:  # If tensor is 2D, it should be (n_mels, length)
        current_length = tensor.size(1)
        current_width = tensor.size(0)
        if current_length < target_length or current_width < target_width:
            padding = (0, target_length - current_length, 0, target_width - current_width)
            tensor = torch.nn.functional.pad(tensor, padding)
    elif tensor.dim() == 3:  # If tensor is 3D, it should be (1, n_mels, length)
        current_length = tensor.size(2)
        current_width = tensor.size(1)
        if current_length < target_length or current_width < target_width:
            padding = (0, target_length - current_length, 0, 0, 0, target_width - current_width)
            tensor = torch.nn.functional.pad(tensor, padding)
    return tensor

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {'input': torch.empty(0), 'target': torch.empty(0)}

    max_length = max(item['input'].size(-1) for item in batch)
    max_width = max(item['input'].size(-2) for item in batch)

    inputs = torch.stack([pad_tensor(item['input'], max_length, max_width).to(torch.float32) for item in batch])
    targets = torch.stack([pad_tensor(item['target'], max_length, max_width).to(torch.float32) for item in batch])
    
    logger.debug(f"Inputs shape after padding: {inputs.shape}")
    logger.debug(f"Targets shape after padding: {targets.shape}")
    
    inputs = inputs.cpu()
    targets = targets.cpu()
    
    return {'input': inputs, 'target': targets}

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
        stop_flag=stop_flag,
        use_cache=True
    )
    dataset.load_all_stems()
    logger.info("Dataset preprocessing and caching complete.")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def preprocess_data(file_path, mel_spectrogram, target_length, apply_data_augmentation, device, cache_dir=None, use_cache=True, extra_features=None):
    """Preprocess and cache individual data files."""
    logger.info(f"Preprocessing file: {file_path}")
    cache_key = os.path.splitext(os.path.basename(file_path))[0]
    cache_file_path = os.path.join(cache_dir, f"{cache_key}.h5") if cache_dir else None

    if use_cache and cache_file_path and os.path.exists(cache_file_path):
        logger.info(f"Loading from cache: {cache_file_path}")
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:])
            target_data = torch.tensor(f['target'][:])
        return input_data, target_data

    data, sample_rate = sf.read(file_path)
    if sample_rate != mel_spectrogram.sample_rate:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=mel_spectrogram.sample_rate)

    input_data = mel_spectrogram(torch.tensor(data).to(device).float()).unsqueeze(0)
    if input_data.shape[-1] < target_length:
        input_data = F.pad(input_data, (0, target_length - input_data.shape[-1]))

    target_data = input_data.clone()

    if extra_features:
        for feature in extra_features:
            feature_data = feature(data, sample_rate).float().to(device)
            target_data = torch.cat((target_data, feature_data), dim=1)

    if use_cache and cache_file_path:
        logger.info(f"Saving to cache: {cache_file_path}")
        with h5py.File(cache_file_path, 'w') as f:
            f.create_dataset('input', data=input_data.cpu().numpy())
            f.create_dataset('target', data=target_data.cpu().numpy())

    return input_data, target_data

def load_and_preprocess(file_path, mel_spectrogram, target_length, device, apply_data_augmentation=False, cache_dir=None, use_cache=True, extra_features=None):
    logger.info(f"Preprocessing file: {file_path}")
    cache_key = os.path.splitext(os.path.basename(file_path))[0]
    cache_file_path = os.path.join(cache_dir, f"{cache_key}.h5") if cache_dir else None

    if use_cache and cache_file_path and os.path.exists(cache_file_path):
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:]).to(device).float()
            target_data = torch.tensor(f['target'][:]).to(device).float()
        logger.info(f"Loaded from cache: {cache_file_path}, input shape: {input_data.shape}, target shape: {target_data.shape}")
        
        # Clear memory after loading cached data
        torch.cuda.empty_cache()  # Clear GPU memory
        import gc
        gc.collect()  # Clear CPU memory

        return input_data, target_data

    try:
        data, sample_rate = sf.read(file_path)
    except Exception as e:  # Add error handling for file reading
        logger.error(f"Error reading file {file_path}: {e}")
        return None, None  

    data = librosa.resample(data, orig_sr=sample_rate, target_sr=mel_spectrogram.sample_rate) if sample_rate != mel_spectrogram.sample_rate else data
    input_mel = mel_spectrogram(torch.tensor(data).to(device).float())

    # Padding is applied here before unsqueezing, ensuring all feature dimensions align correctly
    if input_mel.shape[-1] < target_length:
        input_mel = F.pad(input_mel, (0, target_length - input_mel.shape[-1]))

    input_mel = input_mel.unsqueeze(0)  # Add channel dimension

    logger.info(f"Input data shape after padding: {input_mel.shape}")

    # Initialize target_data without the channel dimension 
    target_data = input_mel.clone()
    
    # Calculate and add extra features to target_data
    if extra_features:
        features = []
        for feature in extra_features:
            feature_data = feature(data, sample_rate, mel_spectrogram.n_mels, target_length).to(device)
            logger.info(f"Feature {feature.__name__} shape before padding: {feature_data.shape}")

            if feature_data.shape[-1] < target_length:
                feature_data = F.pad(feature_data, (0, target_length - feature_data.shape[-1]))

            # Ensure correct dimensions before adding to target_data
            feature_data = feature_data.unsqueeze(0) if feature_data.dim() == 2 else feature_data  # Add channel dimension if needed

            logger.info(f"Feature {feature.__name__} shape after padding: {feature_data.shape}")
            target_data = torch.cat([target_data, feature_data], dim=1)  # Concatenate along the channel dimension
    
    logger.info(f"Target data shape before unsqueeze: {target_data.shape}")
    target_data = target_data.unsqueeze(0)  # Add batch dimension here

    logger.info(f"Target data shape after concatenation: {target_data.shape}")

    if use_cache and cache_file_path:
        with h5py.File(cache_file_path, 'w') as f:
            f.create_dataset('input', data=input_mel.cpu().numpy())
            f.create_dataset('target', data=target_data.cpu().numpy())
        logger.info(f"Saved to cache: {cache_file_path}, input shape: {input_mel.shape}, target shape: {target_data.shape}")

    return input_mel, target_data

def monitor_memory_usage():
    try:
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory_info.percent}%")
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {e}")

def warm_up_cache(dataset):
    dataset.load_all_stems()

def save_batch_to_hdf5(batch, file_path):
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('input', data=batch['input'].cpu().numpy())
            f.create_dataset('target', data=batch['target'].cpu().numpy())
        logger.info(f"Saved batch to {file_path}")
    except Exception as e:
        logger.error(f"Error saving batch to HDF5: {e}")

def load_batch_from_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:]).float()
            target_data = torch.tensor(f['target'][:]).float()
        logger.info(f"Loaded batch from {file_path}")
        return {'input': input_data, 'target': target_data}
    except Exception as e:
        logger.error(f"Error loading batch from HDF5: {e}")
        return None

def dynamic_batching(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for batch in loader:
        yield batch

def check_and_reshape(tensor, target_shape, logger):
    target_numel = target_shape.numel()
    output_numel = tensor.numel()

    if output_numel != target_numel:
        logger.warning(f"Output size ({output_numel}) does not match the expected size ({target_numel}). Attempting to reshape.")
        
        tensor = tensor.view(-1)
        
        if output_numel > target_numel:
            tensor = tensor[:target_numel]
        else:
            padding_size = target_numel - output_numel
            tensor = F.pad(tensor, (0, padding_size))
        
        tensor = tensor.view(target_shape.shape)

    return tensor

def calculate_harmonic_content(data, sample_rate, n_mels, target_length):
    """Calculate harmonic content and match the shape of the Mel spectrogram."""
    harmonic, _ = librosa.effects.hpss(data)
    
    # Ensure harmonic content is 1D (some versions of librosa might return 2D)
    if harmonic.ndim > 1:
        harmonic = harmonic.mean(axis=0)
    
    harmonic_content = torch.tensor(harmonic).float().unsqueeze(0)  # Create tensor and add a dimension
    harmonic_content = pad_tensor(harmonic_content, target_length, n_mels)  # Pad to match Mel spectrogram size
    harmonic_content = harmonic_content[:n_mels, :target_length]  # Crop if necessary
    return harmonic_content.unsqueeze(0)  # Add batch dimension

def calculate_percussive_content(data, sample_rate, n_mels, target_length):
    """Calculate percussive content and match the shape of the Mel spectrogram."""
    _, percussive = librosa.effects.hpss(data)
    
    # Ensure percussive content is 1D
    if percussive.ndim > 1:
        percussive = percussive.mean(axis=0)

    percussive_content = torch.tensor(percussive).float().unsqueeze(0)  # Create tensor and add a dimension
    percussive_content = pad_tensor(percussive_content, target_length, n_mels)
    percussive_content = percussive_content[:n_mels, :target_length]
    return percussive_content.unsqueeze(0)  # Add batch dimension

def load_from_cache(file_path):
    with h5py.File(file_path, 'r') as f:
        input_data = torch.tensor(f['input'][:]).float()
        target_data = torch.tensor(f['target'][:]).float()
    return {'input': input_data, 'target': target_data}

# Add warm_up_cache to the utility functions to ensure cache warming before processing
def warm_up_cache(dataset):
    """Warm up the cache by loading all stems into memory."""
    dataset.load_all_stems()
    logger.info("Cache warming complete.")
