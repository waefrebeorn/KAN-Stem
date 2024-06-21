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
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torchaudio
import h5py
import numpy as np
import librosa
from collections import defaultdict, OrderedDict
from typing import Iterator, List, Dict, Tuple, Any, Union, Callable

logger = logging.getLogger(__name__)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def detect_parameters(data_dir: str, default_n_mels: int = 64, default_n_fft: int = 1024) -> Tuple[int, int, int]:
    sample_rates = []
    durations = []

    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                try:
                    sample_rate, duration, is_silent = analyze_audio(file_path)
                    if is_silent:
                        logger.info(f"Skipping silent file: {file_name}")
                        continue
                    sample_rates.append(sample_rate)
                    durations.append(duration)
                except Exception as e:
                    logger.error(f"Error analyzing audio file {file_path}: {e}")

    logger.info(f"Found {len(sample_rates)} valid audio files")

    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    avg_sample_rate = sum(sample_rates) / len(sample_rates)
    avg_duration = sum(durations) / len(durations)

    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))

    return int(avg_sample_rate), n_mels, n_fft

def analyze_audio(file_path: str) -> Tuple[int, float, bool]:
    try:
        data, sample_rate = sf.read(file_path)
        duration = len(data) / sample_rate
        is_silent = (abs(data).max() < 1e-5)
        return sample_rate, duration, is_silent
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        return None, None, True

def convert_to_3_channels(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.repeat(1, 3, 1, 1)

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, n_ffts: List[int] = [2048, 1024, 512, 256, 128]):
        super(MultiScaleSpectralLoss, self).__init__()
        self.n_ffts = n_ffts

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft in self.n_ffts:
            y_true_mono = y_true[:, 0, :, :]
            y_pred_mono = y_pred[:, 0, :, :]

            if y_true_mono.shape != y_pred_mono.shape:
                raise ValueError(f"Shape mismatch: y_true_mono shape {y_true_mono.shape}, y_pred_mono shape {y_pred_mono.shape}")

            y_true_stft = custom_stft(y_true_mono, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, device=y_true_mono.device)
            y_pred_stft = custom_stft(y_pred_mono, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, device=y_pred_mono.device)

            y_true_mag = torch.abs(y_true_stft)
            y_pred_mag = torch.abs(y_pred_stft)
            y_true_mag = y_true_mag / y_true_mag.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            y_pred_mag = y_pred_mag / y_pred_mag.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            loss += F.l1_loss(y_true_mag, y_pred_mag)
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module, layer_weights: List[float] = None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * len(self.feature_extractor)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        features_true = self.feature_extractor(y_true)
        features_pred = self.feature_extractor(y_pred)
        for weight, feature_true, feature_pred in zip(self.layer_weights, features_true, features_pred):
            loss += weight * F.l1_loss(feature_true, feature_pred)
        return loss

def compute_adversarial_loss(discriminator: nn.Module, y_true: torch.Tensor, y_pred: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    y_true = y_true.unsqueeze(1) if y_true.dim() == 3 else y_true
    y_pred = y_pred.unsqueeze(1) if y_pred.dim() == 3 else y_pred
    real_labels = torch.ones(y_true.size(0), 1, device=device)
    fake_labels = torch.zeros(y_pred.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy_with_logits(discriminator(y_true), real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(y_pred), fake_labels)
    return real_loss, fake_loss

def gradient_penalty(discriminator: nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor, device: torch.device, lambda_gp: float = 10) -> torch.Tensor:
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

def process_file(dataset: 'StemSeparationDataset', file_path: str, target_length: int, apply_data_augmentation: bool, device: torch.device):
    try:
        logger.info(f"Processing file: {file_path}")
        log_system_resources()
        start_time = time.time()

        cache_key = dataset._get_cache_key(file_path, apply_data_augmentation)
        cached_data = dataset._get_data(file_path, apply_data_augmentation)
        
        if cached_data is not None:
            logger.info(f"Loaded from cache: {file_path}")
            return

        input_mel, target_mel = load_and_preprocess(
            dataset, file_path, target_length, device, apply_data_augmentation=apply_data_augmentation,
            cache_dir=dataset.cache_dir, use_cache=dataset.use_cache, extra_features=[calculate_harmonic_content, calculate_percussive_content]
        )

        if input_mel is not None and target_mel is not None:
            data = {"input": input_mel, "target": target_mel}
            dataset._save_individual_cache(file_path, data, apply_data_augmentation)

        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")

        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        raise e

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    try:
        mse = F.mse_loss(y_pred, y_true).item()
        snr = 10 * torch.log10(torch.mean(y_true ** 2) / torch.mean((y_true - y_pred) ** 2)).item()
        return {'mse': mse, 'snr': snr}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        raise e

def get_checkpoints(checkpoints_dir: str) -> List[str]:
    try:
        checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints
    except Exception as e:
        logger.error(f"Error retrieving checkpoints: {e}")
        return []

def compute_sdr(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    sdr = 10 * torch.log10(s_true / (s_noise + 1e-8))
    return sdr

def compute_sir(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_interf = torch.sum((true - noise) ** 2, dim=[1, 2, 3])
    sir = 10 * torch.log10(s_true / (s_interf + 1e-8))
    return sir

def compute_sar(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    noise = true - pred
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    s_artif = torch.sum((pred - noise) ** 2, dim=[1, 2, 3])
    sar = 10 * torch.log10(s_noise / (s_artif + 1e-8))
    return sar

def log_training_parameters(params: Dict[str, Any]):
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

def get_optimizer(optimizer_name: str, model_parameters: Any, learning_rate: float, weight_decay: float) -> optim.Optimizer:
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

def wasserstein_loss(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    return torch.mean(fake_output) - torch.mean(real_output)

class GPULRUCache:
    def __init__(self, capacity: int, device: torch.device):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.device = device

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                _, evicted_value = self.cache.popitem(last=False)
                if isinstance(evicted_value, torch.Tensor):
                    del evicted_value  # Explicitly delete the tensor
        self.cache[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        torch.cuda.empty_cache()  # Clear unused memory

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def clear(self):
        self.cache.clear()
        torch.cuda.empty_cache()

def scan_cache_directory(cache_dir: str) -> Dict[str, str]:
    cache_index = {}
    for root, _, files in os.walk(cache_dir):
        for file_name in files:
            if file_name.endswith('.h5'):
                cache_key = os.path.splitext(file_name)[0]
                file_path = os.path.join(root, file_name)
                cache_index[cache_key] = file_path
    return cache_index

@torch.cuda.amp.autocast()
def load_from_cache(file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        logger.info(f"Attempting to load cache file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Cache file does not exist: {file_path}")
            raise FileNotFoundError(f"Cache file does not exist: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            input_data = torch.from_numpy(f['input'][:]).to(device, non_blocking=True).float()
            target_data = torch.from_numpy(f['target'][:]).to(device, non_blocking=True).float()
        
        logger.info(f"Successfully loaded cache file: {file_path}")
        logger.info(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")
        
        return {
            'input': input_data,
            'target': target_data
        }
    except Exception as e:
        logger.error(f"Error loading from cache file '{file_path}': {e}")
        logger.error(f"File exists: {os.path.exists(file_path)}")
        logger.error(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise

def is_cache_warmed_up(cache_dir):
    required_files = ['input.h5', 'target.h5']
    for file in required_files:
        if not os.path.exists(os.path.join(cache_dir, file)):
            return False
    return True

def warm_up_cache(cache_dir):
    if not is_cache_warmed_up(cache_dir):
        # Assume this function will generate the cache files required
        generate_cache_files(cache_dir)
    logger.info(f"Cache warmed up for: {cache_dir}")

class StemSeparationDataset(Dataset):
    def __init__(
        self, data_dir: str, n_mels: int, target_length: int, n_fft: int, cache_dir: str, apply_data_augmentation: bool,
        suppress_warnings: bool, suppress_reading_messages: bool, num_workers: int, device_prep: torch.device, stop_flag: Any,
        use_cache: bool = True, cache_capacity: int = 100, mel_spectrogram: T.MelSpectrogram = None, stem_names: Dict[str, List[str]] = None
    ):
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
        self.training = True  # Default to True

        if mel_spectrogram is None:
            self.mel_spectrogram = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=n_fft,
                win_length=None,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                power=2.0
            ).to(torch.float32).to(device_prep)
        else:
            self.mel_spectrogram = mel_spectrogram

        if stem_names is None:
            self.stem_names = self._load_stem_names()
        else:
            self.stem_names = stem_names

        if not self.stem_names:
            raise ValueError(f"No valid audio files found in {data_dir}")

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index = self._load_cache_metadata()

        self.cache = GPULRUCache(cache_capacity, device_prep)

    def _load_stem_names(self) -> Dict[str, List[str]]:
        stem_names = defaultdict(list)
        for root, _, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.wav'):
                    parts = file_name.split('_')
                    stem_name = parts[1] if parts[0] == 'target' else "input"
                    file_path = os.path.join(root, file_name)
                    stem_names[stem_name].append(file_path)
        return stem_names

    def _load_cache_metadata(self) -> Dict[str, str]:
        index_file_path = os.path.join(self.cache_dir, "cache_index.h5")
        cache_index = {}
        if os.path.exists(index_file_path):
            try:
                with h5py.File(index_file_path, 'r') as f:
                    for key in f.attrs:
                        cache_index[key] = f.attrs[key]
                if not self.suppress_reading_messages:
                    logger.info(f"Loaded cache metadata from {index_file_path}")
            except Exception as e:
                if not self.suppress_reading_messages:
                    logger.error(f"Error loading cache metadata file: {e}")

        existing_cache_files = scan_cache_directory(self.cache_dir)
        cache_index.update(existing_cache_files)
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

    def __len__(self) -> int:
        return len(self.stem_names["input"])  # count all files in "input" stem

    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], None]:
        if idx >= len(self):
            return None

        file_name = self.stem_names["input"][idx]  # Get input file name directly
        apply_augmentation = self.apply_data_augmentation if self.training else False 
        
        try:
            data = self._get_data(file_name, apply_augmentation)
            if data is None:
                logger.error(f"Failed to load data for {file_name} from cache (augmented={apply_augmentation})")
                return None
            data['file_path'] = file_name  # Add original file_path to data
            return data
        except Exception as e:
            logger.error(f"Error in __getitem__ for stem {file_name}: {e}")
            return None

    def _get_cache_key(self, stem_name: str, identifier: str, apply_data_augmentation: bool) -> str:
        return f"{stem_name}_{identifier}_{apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"
    
    def _get_data(self, file_name: str, apply_data_augmentation: bool) -> Union[Dict[str, torch.Tensor], None]:
        identifier = file_name.split('_')[1].split('.')[0]
        stem_name = file_name.split('_')[0]

        cache_key = self._get_cache_key(stem_name, identifier, apply_data_augmentation)
        data = self.cache.get(cache_key)
        if data is not None:
            return data

        cached_data_path = os.path.join(self.cache_dir, f"{stem_name}_{identifier}_{apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}.h5")

        if self.use_cache and cached_data_path and os.path.exists(cached_data_path):
            try:
                data = load_from_cache(cached_data_path, self.device_prep)
                if self._validate_data(data):
                    if not self.suppress_reading_messages:
                        logger.info(f"Loaded cached data for {file_name} (augmented={apply_data_augmentation}) from {cached_data_path}")
                    data['input'] = data['input'].to(self.device_prep).float()
                    data['target'] = data['target'].to(self.device_prep).float()
                    self.cache.put(cache_key, data)
                    return data
                else:
                    if not self.suppress_reading_messages:
                        logger.warning(f"Invalid cached data for {file_name} (augmented={apply_data_augmentation}). Reprocessing.")
            except (FileNotFoundError, KeyError) as e:
                if not self.suppress_reading_messages:
                    logger.warning(f"Error loading cached data for {file_name} (augmented={apply_data_augmentation}). Reprocessing. Error: {e}")

        data = self._process_and_cache(file_name, apply_data_augmentation)
        if data is not None:
            self.cache.put(cache_key, data)
        return data

    def _process_and_cache(self, file_name: str, apply_data_augmentation: bool) -> Dict[str, torch.Tensor]:
        if self.stop_flag.value == 1:
            return None

        try:
            if file_name.startswith("input"):
                file_path = self.stem_names["input"][int(file_name.split('_')[1].split('.')[0]) - 1]
            else:
                file_path = os.path.join(self.cache_dir, file_name)
                
            logger.info(f"Processing file: {file_path}")

            input_mel, target_mel = load_and_preprocess(
                self, file_path, self.target_length, self.device_prep,
                cache_dir=self.cache_dir, use_cache=self.use_cache, extra_features=[
                    calculate_harmonic_content,
                    calculate_percussive_content
                ]
            )

            if input_mel is None:
                raise ValueError(f"Failed to load and preprocess data for {file_name}")

            input_mel = input_mel.to(torch.float32)
            target_mel = target_mel.to(torch.float32)

            input_mel = input_mel.unsqueeze(0)

            if input_mel.shape[-1] < self.target_length:
                input_mel = torch.nn.functional.pad(input_mel, (0, self.target_length - input_mel.shape[-1]), mode='constant')

            if input_mel.shape[0] != target_mel.shape[0]:
                input_mel = input_mel.unsqueeze(0)
            if target_mel.shape[0] != input_mel.shape[0]:
                target_mel = target_mel.unsqueeze(0)

            data = {"input": input_mel, "target": target_mel}

            self._save_individual_cache(file_name, data, apply_data_augmentation) 

            return data
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error processing stem {file_name}: {e}")
            return None

    def _save_individual_cache(self, file_name: str, data: Dict[str, torch.Tensor], apply_data_augmentation: bool) -> Union[str, None]:
        identifier = file_name.split('_')[1].split('.')[0]
        stem_name = file_name.split('_')[0]  # Extract stem name (e.g., 'target_bass')

        cache_file_name = f"{stem_name}_{identifier}_{apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}.h5"  # Construct cache filename using stem_name
        stem_cache_path = os.path.join(self.cache_dir, cache_file_name)
        try:
            if not self.suppress_reading_messages:
                logger.info(f"Saving stem cache: {stem_cache_path}")
            with h5py.File(stem_cache_path, 'w') as f:
                f.create_dataset('input', data=data['input'].cpu().detach().numpy())
                f.create_dataset('target', data=data['target'].cpu().detach().numpy())
            if not self.suppress_reading_messages:
                logger.info(f"Successfully saved stem cache: {stem_cache_path}")
            self.cache_index[cache_file_name] = stem_cache_path
            self._save_cache_metadata()
            return stem_cache_path
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error saving stem cache: {stem_cache_path}. Error: {e}")
            return None

    def _validate_data(self, data: Dict[str, torch.Tensor]) -> bool:
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
        expected_input_shape = (1, self.n_mels, self.target_length)
        expected_target_shape = (1, 3, self.n_mels, self.target_length)

        if input_shape != expected_input_shape:
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: Input shape mismatch. Input shape: {input_shape}, Expected input shape: {expected_input_shape}")
            return False

        if target_shape != expected_target_shape:
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: Target shape mismatch. Target shape: {target_shape}, Expected target shape: {expected_target_shape}")
            return False

        return True

    def apply_augmentation(self, input_mel: torch.Tensor, target_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(input_mel, device=self.device_prep) * 0.1
        augmented_input = input_mel + noise
        augmented_target = target_mel + noise
        return augmented_input, augmented_target

def pad_tensor(tensor: torch.Tensor, target_length: int, target_width: int) -> torch.Tensor:
    if tensor.dim() >= 2:
        current_length = tensor.size(1)
        current_width = tensor.size(0)
        if current_length < target_length or current_width < target_width:
            padding = (0, target_length - current_length, 0, target_width - current_width)
            tensor = torch.nn.functional.pad(tensor, padding)
    return tensor

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return {'input': torch.empty(0), 'target': torch.empty(0), 'file_paths': []}

    max_length = max(item['input'].size(-1) for item in batch)
    max_width = max(item['input'].size(-2) for item in batch)

    inputs = torch.stack([pad_tensor(item['input'], max_length, max_width).to(torch.float32) for item in batch])
    targets = torch.stack([pad_tensor(item['target'], max_length, max_width).to(torch.float32) for item in batch])

    logger.debug(f"Inputs shape after padding: {inputs.shape}")
    logger.debug(f"Targets shape after padding: {targets.shape}")

    inputs = inputs.cpu()
    targets = targets.cpu()

    file_paths = [item['file_path'] for item in batch]

    return {'input': inputs, 'target': targets, 'file_paths': file_paths}

def preprocess_and_cache_dataset(
    data_dir: str,
    n_mels: int,
    target_length: int,
    n_fft: int,
    cache_dir: str,
    apply_data_augmentation: bool,
    suppress_warnings: bool,
    suppress_reading_messages: bool,
    num_workers: int,
    device_prep: torch.device,
    stop_flag: Any,
    validation_split: float = 0.1,
    random_seed=42
) -> Tuple[StemSeparationDataset, StemSeparationDataset]:

    # Load stem names
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=22050,
        n_fft=n_fft,
        win_length=None,
        hop_length=n_fft // 4,
        n_mels=n_mels,
        power=2.0
    ).to(torch.float32).to(device_prep)

    # Create initial dataset to get all file names
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
        use_cache=True,
        mel_spectrogram=mel_spectrogram
    )

    # Process files if cache doesn't exist
    for file_name in dataset.stem_names['input']:
        process_file(dataset, file_name, target_length, apply_data_augmentation, device_prep)

    # Get all stem names
    stem_names = dataset._load_stem_names()

    # Load and split file names for train and validation
    np.random.seed(random_seed)
    all_input_files = [f for f in os.listdir(cache_dir) if f.endswith('.h5') and f.startswith('input')]
    np.random.shuffle(all_input_files)
    split_idx = int(len(all_input_files) * validation_split)
    validation_input_files = all_input_files[:split_idx]
    train_input_files = all_input_files[split_idx:]

    # Extract unique identifiers for all input files
    all_identifiers = {f.split('_')[1].split('.')[0] for f in all_input_files}
    train_identifiers = {f.split('_')[1].split('.')[0] for f in train_input_files}
    validation_identifiers = {f.split('_')[1].split('.')[0] for f in validation_input_files}

    # Create dictionaries to store train and validation stem names
    train_stem_names = defaultdict(list)
    validation_stem_names = defaultdict(list)
    for stem_name, file_paths in stem_names.items():
        if stem_name == "input":
            train_stem_names[stem_name] = train_input_files
            validation_stem_names[stem_name] = validation_input_files
        else:
            train_stem_names[stem_name] = [f for f in file_paths if any(id in f for id in train_identifiers)]
            validation_stem_names[stem_name] = [f for f in file_paths if any(id in f for id in validation_identifiers)]

    # Create train and validation datasets
    train_dataset = StemSeparationDataset(
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
        use_cache=True,
        mel_spectrogram=mel_spectrogram,
        stem_names=train_stem_names
    )
    train_dataset.training = True  # Set training to True

    validation_dataset = StemSeparationDataset(
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
        use_cache=True,
        mel_spectrogram=mel_spectrogram,
        stem_names=validation_stem_names
    )
    validation_dataset.training = False  # Set training to False

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(validation_dataset)}")

    return train_dataset, validation_dataset

def ensure_dir_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_and_preprocess(
    dataset: 'StemSeparationDataset', file_path: str, target_length: int, device: torch.device,
    apply_data_augmentation: bool = False, cache_dir: str = None, use_cache: bool = True, extra_features: List[Callable] = None,
    suppress_messages: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not suppress_messages:
        logger.info(f"Preprocessing file: {file_path}")
    cache_key = os.path.splitext(os.path.basename(file_path))[0]
    cache_file_path = os.path.join(cache_dir, f"{cache_key}.h5") if cache_dir else None

    if use_cache and cache_file_path and os.path.exists(cache_file_path):
        if not suppress_messages:
            logger.info(f"Loading from cache: {cache_file_path}")
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:]).to(device).float()
            target_data = torch.tensor(f['target'][:]).to(device).float()
        if not suppress_messages:
            logger.info(f"Loaded from cache: {cache_file_path}, input data shape: {input_data.shape}, target data shape: {target_data.shape}")
        return input_data, target_data

    try:
        data, sample_rate = sf.read(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None, None

    try:
        if data.ndim == 2:
            data = data.mean(axis=1)

        if sample_rate != dataset.mel_spectrogram.sample_rate:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=dataset.mel_spectrogram.sample_rate)

        data_tensor = torch.tensor(data).to(device).float().unsqueeze(0)

        input_mel = dataset.mel_spectrogram(waveform=data_tensor)
        if input_mel.shape[-1] < target_length:
            input_mel = F.pad(input_mel, (0, target_length - input_mel.shape[-1]))

        # Ensure input_mel is 3D (channels, n_mels, target_length)
        input_mel = input_mel.squeeze(0)  # Remove batch dimension if present

        if not suppress_messages:
            logger.info(f"Input data shape after processing: {input_mel.shape}")

        features = []
        if extra_features:
            for feature in extra_features:
                feature_data = feature(input_mel, sample_rate, dataset.mel_spectrogram, dataset.n_mels, target_length).to(device).float()
                if feature_data.shape[-1] < target_length:
                    feature_data = F.pad(feature_data, (0, target_length - feature_data.shape[-1]))
                features.append(feature_data)

        target_data = torch.stack([input_mel] + features, dim=0)

        if not suppress_messages:
            logger.info(f"Target data shape after stacking: {target_data.shape}")

        if apply_data_augmentation:
            _, target_data = dataset.apply_augmentation(input_mel, target_data)  # Only augment target

        if use_cache and cache_file_path:
            with h5py.File(cache_file_path, 'w') as f:
                f.create_dataset('input', data=input_mel.cpu().numpy())
                f.create_dataset('target', data=target_data.cpu().numpy())
            if not suppress_messages:
                logger.info(f"Saved to cache: {cache_file_path}, input shape: {input_mel.shape}, target shape: {target_data.shape}")

        return input_mel, target_data

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None, None
        
def monitor_memory_usage():
    try:
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory_info.percent}%")
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {e}")

def save_batch_to_hdf5(batch: Dict[str, torch.Tensor], file_path: str):
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('input', data=batch['input'].cpu().numpy())
            f.create_dataset('target', data=batch['target'].cpu().numpy())
        logger.info(f"Saved batch to {file_path}")
    except Exception as e:
        logger.error(f"Error saving batch to HDF5: {e}")

def load_batch_from_hdf5(file_path: str) -> Union[Dict[str, torch.Tensor], None]:
    try:
        with h5py.File(file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:]).float()
            target_data = torch.tensor(f['target'][:]).float()
        logger.info(f"Loaded batch from {file_path}")
        return {'input': input_data, 'target': target_data}
    except (FileNotFoundError, h5py.H5Error, KeyError) as e:
        logger.error(f"Error loading from cache file '{file_path}': {e}")
        raise

def integrated_dynamic_batching(dataset: StemSeparationDataset, batch_size: int, stem_name: str, shuffle: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Integrates dynamic batching with cache warming for efficient data loading.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    for batch in dataloader:
        warmed_inputs = []
        warmed_targets = []
        warmed_file_paths = []

        for i, file_path in enumerate(batch['file_paths']):
            identifier = file_path.split('_')[1].split('.')[0]
            input_cache_file_name = f"input_{identifier}.h5"
            target_cache_file_name = f"target_{stem_name}_{identifier}.h5"
            
            # Use os.path.join to create the full path
            input_cache_file_path = os.path.join(dataset.cache_dir, input_cache_file_name)
            target_cache_file_path = os.path.join(dataset.cache_dir, target_cache_file_name)

            if os.path.exists(input_cache_file_path) and os.path.exists(target_cache_file_path):
                try:
                    input_data = load_from_cache(input_cache_file_path, dataset.device_prep)
                    target_data = load_from_cache(target_cache_file_path, dataset.device_prep)
                    
                    warmed_inputs.append(input_data['input'])
                    warmed_targets.append(target_data['target'])
                    warmed_file_paths.append(file_path)
                    
                    logger.info(f"Loaded and warmed up cache for {file_path}")
                except Exception as e:
                    logger.error(f"Error loading cache for {file_path}: {e}")
            else:
                logger.warning(f"Cache files not found for {file_path}. Skipping.")

        if warmed_inputs and warmed_targets:
            yield {
                'inputs': torch.stack(warmed_inputs),
                'targets': torch.stack(warmed_targets),
                'file_paths': warmed_file_paths
            }

def check_and_reshape(tensor: torch.Tensor, target_shape: torch.Tensor, logger: logging.Logger) -> torch.Tensor:
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

def calculate_harmonic_content(mel_spectrogram: torch.Tensor, sample_rate: int, mel_spectrogram_object: T.MelSpectrogram, n_mels: int, target_length: int) -> torch.Tensor:
    harmonic_content = mel_spectrogram.squeeze()[:, :n_mels]  

    harmonic_content = torchaudio.transforms.AmplitudeToDB()(harmonic_content)
    harmonic_content = pad_tensor(harmonic_content, target_length, n_mels)
    harmonic_content = harmonic_content[:n_mels, :target_length]

    return harmonic_content.to(torch.float32)

def calculate_percussive_content(mel_spectrogram: torch.Tensor, sample_rate: int, mel_spectrogram_object: T.MelSpectrogram, n_mels: int, target_length: int) -> torch.Tensor:
    percussive_content = mel_spectrogram.squeeze()[:, n_mels // 2:]
    percussive_content = torchaudio.transforms.AmplitudeToDB()(percussive_content)

    percussive_content = pad_tensor(percussive_content, target_length, n_mels)
    percussive_content = percussive_content[:n_mels, :target_length]

    return percussive_content.to(torch.float32)

def purge_vram():
    """
    Purge the GPU cache to free up memory.
    """
    try:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Successfully purged GPU cache and collected garbage.")
    except Exception as e:
        logger.error(f"Error purging cache: {e}", exc_info=True)
