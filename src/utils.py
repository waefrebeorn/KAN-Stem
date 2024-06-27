import os
import torch
import torch.nn.functional as F
import logging
import soundfile as sf
import h5py
import numpy as np
import librosa
from collections import defaultdict
from typing import Dict, Tuple, Union, List, Any
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torch.optim as optim
import torch.nn as nn

logger = logging.getLogger(__name__)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def detect_parameters(data_dir: str, default_n_mels: int = 64, default_n_fft: int = 1024) -> Tuple[int, int, int]:
    sample_rates, durations = [], []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                try:
                    info = sf.info(file_path)
                    sample_rates.append(info.samplerate)
                    durations.append(info.duration)
                except Exception as e:
                    logger.error(f"Error analyzing audio file {file_path}: {e}")

    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    logger.info(f"Found {len(sample_rates)} valid audio files")
    avg_sample_rate = int(np.mean(sample_rates))
    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))
    return avg_sample_rate, n_mels, n_fft

def resize_tensor(tensor: torch.Tensor, target_freq: int, target_length: int) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[-2] != target_freq:
        tensor = F.interpolate(tensor, size=(target_freq, tensor.shape[-1]), mode='nearest')

    if tensor.shape[-1] < target_length:
        tensor = F.pad(tensor, (0, target_length - tensor.shape[-1]))
    elif tensor.shape[-1] > target_length:
        tensor = tensor[..., :target_length]

    return tensor

def calculate_harmonic_content(mel_spectrogram: torch.Tensor, sample_rate: int, mel_spectrogram_object: T.MelSpectrogram, n_mels: int, target_length: int) -> torch.Tensor:
    harmonic_content = mel_spectrogram[:, :, :n_mels, :]

    amplitude_to_db = T.AmplitudeToDB()
    harmonic_content = torch.stack([amplitude_to_db(harmonic_content[:, i]) for i in range(harmonic_content.shape[1])], dim=1)

    harmonic_content = resize_tensor(harmonic_content, n_mels, target_length)

    return harmonic_content.to(torch.float32)

def calculate_percussive_content(mel_spectrogram: torch.Tensor, sample_rate: int, mel_spectrogram_object: T.MelSpectrogram, n_mels: int, target_length: int) -> torch.Tensor:
    percussive_content = mel_spectrogram[:, :, n_mels // 2:, :]

    amplitude_to_db = T.AmplitudeToDB()
    percussive_content = torch.stack([amplitude_to_db(percussive_content[:, i]) for i in range(percussive_content.shape[1])], dim=1)

    percussive_content = resize_tensor(percussive_content, n_mels, target_length)

    return percussive_content.to(torch.float32)

class StemSeparationDataset(Dataset):
    def __init__(
        self, data_dir: str, n_mels: int, target_length: int, n_fft: int, cache_dir: str,
        apply_data_augmentation: bool, device: torch.device, suppress_warnings: bool = False,
        suppress_reading_messages: bool = False, num_workers: int = 1, stem_names: List[str] = None,
        stop_flag: Any = None, use_cache: bool = True, device_prep: torch.device = None, segments_per_track: int = 10
    ):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.device = device
        self.suppress_warnings = suppress_warnings
        self.suppress_reading_messages = suppress_reading_messages
        self.num_workers = num_workers
        self.stop_flag = stop_flag
        self.use_cache = use_cache
        self.training = True
        self.device_prep = device_prep or device
        self.segments_per_track = segments_per_track

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=22050, n_fft=n_fft, n_mels=n_mels,
            win_length=None, hop_length=n_fft // 4, power=2.0
        ).to(self.device_prep)
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device_prep)

        self.stem_names = stem_names or ["vocals", "other", "noise", "keys", "guitar", "drums", "bass"]
        self.file_ids = self._get_file_ids()
        os.makedirs(cache_dir, exist_ok=True)

    def _get_file_ids(self):
        file_ids = []
        for input_file in os.listdir(self.data_dir):
            if input_file.startswith('input_') and input_file.endswith('.wav'):
                identifier = input_file.split('_')[1].split('.')[0]
                target_files = {
                    stem: f"target_{stem}_{identifier}.wav" for stem in self.stem_names
                }
                file_ids.append({
                    'identifier': identifier,
                    'input_file': input_file,
                    'target_files': target_files
                })
        return file_ids

    def _get_cache_path(self, stem_name: str, identifier: str, augmented: bool) -> str:
        return os.path.join(self.cache_dir, f"{stem_name}_{identifier}_{str(augmented)}_{self.n_mels}_{self.target_length}_{self.n_fft}.h5")

    def _load_or_process_file(self, file_path: str, stem_name: str, augmented: bool) -> torch.Tensor:
        identifier = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
        cache_path = self._get_cache_path(stem_name, identifier, augmented)

        if os.path.exists(cache_path):
            with h5py.File(cache_path, 'r') as f:
                return torch.from_numpy(f['data'][:]).to(self.device)

        data, sr = sf.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 22050:
            data = librosa.resample(data, orig_sr=sr, target_sr=22050)

        mel_spec = self.mel_spectrogram(torch.tensor(data).float().to(self.device).unsqueeze(0))
        mel_spec = self.amplitude_to_db(mel_spec)

        mel_spec = resize_tensor(mel_spec, self.n_mels, self.target_length)

        harmonic_content = calculate_harmonic_content(mel_spec, 22050, self.mel_spectrogram, self.n_mels, self.target_length)
        percussive_content = calculate_percussive_content(mel_spec, 22050, self.mel_spectrogram, self.n_mels, self.target_length)

        combined_spec = torch.cat([mel_spec, harmonic_content, percussive_content], dim=1)

        if augmented:
            combined_spec = self.apply_augmentation(combined_spec)

        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('data', data=combined_spec.cpu().numpy())

        return combined_spec

    def apply_augmentation(self, mel_spec: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(mel_spec) * 0.005
        return mel_spec + noise

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.file_ids[idx]
        identifier = file_id['identifier']
        
        input_cache = self._get_cache_path('input', identifier, True)
        input_data = self._load_or_process_if_needed(input_cache, file_id['input_file'], 'input', False)
        
        target_data = {}
        for stem in self.stem_names:
            target_cache_true = self._get_cache_path(stem, identifier, False)
            target_cache_false = self._get_cache_path(stem, identifier, True)
            target_data[stem] = {
                'true': self._load_or_process_if_needed(target_cache_true, file_id['target_files'][stem], stem, False),
                'false': self._load_or_process_if_needed(target_cache_false, file_id['target_files'][stem], stem, True)
            }
        
        return {"input": input_data, "target": target_data, "file_id": identifier}

    def _load_or_process_if_needed(self, cache_path: str, file_path: str, stem_name: str, augmented: bool) -> torch.Tensor:
        if not os.path.exists(cache_path):
            full_file_path = os.path.join(self.data_dir, file_path)
            return self._load_or_process_file(full_file_path, stem_name, augmented)
        return self._load_from_cache(cache_path)

    def _load_from_cache(self, cache_path: str) -> torch.Tensor:
        with h5py.File(cache_path, 'r') as f:
            return torch.from_numpy(f['data'][:]).to(self.device)

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    inputs = torch.cat([item['input'] for item in batch], dim=0)
    
    targets = {}
    for item in batch:
        for stem, target_dict in item['target'].items():
            if stem not in targets:
                targets[stem] = {'true': [], 'false': []}
            targets[stem]['true'].append(target_dict['true'])
            targets[stem]['false'].append(target_dict['false'])
    
    for stem in targets:
        targets[stem]['true'] = torch.cat(targets[stem]['true'], dim=0)
        targets[stem]['false'] = torch.cat(targets[stem]['false'], dim=0)
    
    file_ids = [item['file_id'] for item in batch]
    
    return {'input': inputs, 'target': targets, 'file_ids': file_ids}

def create_dataloader(dataset: StemSeparationDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def process_file(dataset: StemSeparationDataset, file_path: str, target_length: int, apply_data_augmentation: bool, device: torch.device):
    identifier = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
    
    input_cache_path = dataset._get_cache_path('input', identifier, True)
    if not os.path.exists(input_cache_path):
        dataset._load_or_process_file(file_path, 'input', True)

    for stem in dataset.stem_names:
        target_cache_path_true = dataset._get_cache_path(stem, identifier, True)
        target_cache_path_false = dataset._get_cache_path(stem, identifier, False)
        if not os.path.exists(target_cache_path_true):
            dataset._load_or_process_file(file_path, stem, False)
        if not os.path.exists(target_cache_path_false):
            dataset._load_or_process_file(file_path, stem, True)

def preprocess_and_cache_dataset(
    data_dir: str,
    n_mels: int,
    target_length: int,
    n_fft: int,
    cache_dir: str,
    apply_data_augmentation: bool,
    device: torch.device,
    validation_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[StemSeparationDataset, StemSeparationDataset]:
    np.random.seed(random_seed)
    all_input_files = [f for f in os.listdir(data_dir) if f.endswith('.wav') and f.startswith('input')]
    np.random.shuffle(all_input_files)
    split_idx = int(len(all_input_files) * validation_split)
    validation_input_files = all_input_files[:split_idx]
    train_input_files = all_input_files[split_idx:]

    train_identifiers = {f.split('_')[1].split('.')[0] for f in train_input_files}
    validation_identifiers = {f.split('_')[1].split('.')[0] for f in validation_input_files}

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, device)
    stem_names = dataset.stem_names

    train_stem_names = defaultdict(list)
    validation_stem_names = defaultdict(list)
    for stem_name in stem_names:
        if stem_name == "input":
            train_stem_names[stem_name] = train_input_files
            validation_stem_names[stem_name] = validation_input_files
        else:
            train_stem_names[stem_name] = [f for f in os.listdir(data_dir) if f.startswith(stem_name) and f.split('_')[1].split('.')[0] in train_identifiers]
            validation_stem_names[stem_name] = [f for f in os.listdir(data_dir) if f.startswith(stem_name) and f.split('_')[1].split('.')[0] in validation_identifiers]

    train_dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, device, stem_names=train_stem_names)
    train_dataset.training = True

    validation_dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, device, stem_names=validation_stem_names)
    validation_dataset.training = False

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(validation_dataset)}")

    return train_dataset, validation_dataset

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

def purge_vram():
    try:
        torch.cuda.empty_cache()
        logger.info("Successfully purged GPU cache.")
    except Exception as e:
        logger.error(f"Error purging cache: {e}", exc_info=True)

def load_from_cache(file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        logger.info(f"Attempting to load cache file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Cache file does not exist: {file_path}")
            raise FileNotFoundError(f"Cache file does not exist: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            data = torch.from_numpy(f['data'][:]).to(device, non_blocking=True).float()
        
        logger.info(f"Successfully loaded cache file: {file_path}")
        logger.info(f"Data shape: {data.shape}")
        
        return {
            'data': data,
        }
    except Exception as e:
        logger.error(f"Error loading from cache file '{file_path}': {e}")
        logger.error(f"File exists: {os.path.exists(file_path)}")
        logger.error(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise
        
def integrated_dynamic_batching(dataset: StemSeparationDataset, batch_size: int, stem_name: str, shuffle: bool = True) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    for batch in dataloader:
        warmed_inputs = []
        warmed_targets = []
        warmed_file_paths = []

        for i, file_path in enumerate(batch['file_ids']):
            identifier = file_path
            input_cache_file_name = f"input_{identifier}_True_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
            target_cache_file_name = f"{stem_name}_{identifier}_False_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
            
            input_cache_file_path = os.path.join(dataset.cache_dir, input_cache_file_name)
            target_cache_file_path = os.path.join(dataset.cache_dir, target_cache_file_name)

            if os.path.exists(input_cache_file_path) and os.path.exists(target_cache_file_path):
                try:
                    input_data = load_from_cache(input_cache_file_path, dataset.device_prep)
                    target_data = load_from_cache(target_cache_file_path, dataset.device_prep)
                    
                    warmed_inputs.append(input_data['data'])
                    warmed_targets.append(target_data['data'])
                    warmed_file_paths.append(file_path)
                    
                    logger.info(f"Loaded and warmed up cache for {file_path}")
                except Exception as e:
                    logger.error(f"Error loading cache for {file_path}: {e}")
            else:
                logger.warning(f"Cache files not found for {file_path}. Processing now.")
                input_file_path = os.path.join(dataset.data_dir, f'input_{identifier}.wav')
                process_file(dataset, input_file_path, dataset.target_length, apply_data_augmentation=False, device=dataset.device_prep)

                for stem in dataset.stem_names:
                    target_file_path = os.path.join(dataset.data_dir, f'target_{stem}_{identifier}.wav')
                    process_file(dataset, target_file_path, dataset.target_length, apply_data_augmentation=True, device=dataset.device_prep)

                if os.path.exists(input_cache_file_path) and os.path.exists(target_cache_file_path):
                    try:
                        input_data = load_from_cache(input_cache_file_path, dataset.device_prep)
                        target_data = load_from_cache(target_cache_file_path, dataset.device_prep)
                        
                        warmed_inputs.append(input_data['data'])
                        warmed_targets.append(target_data['data'])
                        warmed_file_paths.append(file_path)
                    except Exception as e:
                        logger.error(f"Error loading cache for {file_path}: {e}")

        if warmed_inputs and warmed_targets:
            yield {
                'inputs': torch.stack(warmed_inputs),
                'targets': torch.stack(warmed_targets),
                'file_paths': warmed_file_paths
            }

def ensure_dir_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
