import os
import torch
import torch.nn.functional as F
import logging
import soundfile as sf
import h5py
import numpy as np
import librosa
from typing import Dict, Tuple, Union, List, Any
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.optim as optim
import torch.nn as nn  # Make sure this is imported
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def detect_parameters_from_cache(cache_dir: str, default_n_mels: int = 32, default_n_fft: int = 1024) -> Tuple[int, int, int]:
    sample_rates = []
    for file_name in os.listdir(cache_dir):
        if file_name.endswith('.h5'):
            file_path = os.path.join(cache_dir, file_name)
            try:
                with h5py.File(file_path, 'r') as f:
                    sample_rates.append(f.attrs.get('sample_rate', 44100))
            except Exception as e:
                logger.error(f"Error analyzing cache file {file_path}: {e}")
    
    if not sample_rates:
        raise ValueError("No valid cache files found in the specified directory")

    logger.info(f"Found {len(sample_rates)} valid cache files")
    avg_sample_rate = int(np.mean(sample_rates))
    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))
    return avg_sample_rate, n_mels, n_fft

def detect_parameters_from_raw_data(data_dir: str, default_n_mels: int = 32, default_n_fft: int = 1024) -> Tuple[int, int, int]:
    sample_rates = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith('.ogg'):
                file_path = os.path.join(root, file_name)
                try:
                    info = sf.info(file_path)
                    sample_rates.append(info.samplerate)
                except Exception as e:
                    logger.error(f"Error analyzing audio file {file_path}: {e}")

    if not sample_rates:
        raise ValueError("No valid raw audio files found in the specified directory")

    logger.info(f"Found {len(sample_rates)} valid raw audio files")
    avg_sample_rate = int(np.mean(sample_rates))
    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))
    return avg_sample_rate, n_mels, n_fft

def segment_audio(audio: np.ndarray, chunk_size: int = 22050) -> List[np.ndarray]:
    num_chunks = (len(audio) + chunk_size - 1) // chunk_size
    chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    if len(chunks[-1]) < chunk_size:
        chunks[-1] = np.pad(chunks[-1], (0, chunk_size - len(chunks[-1])), 'constant')
    return chunks

class StemSeparationDataset(Dataset):
    def __init__(
        self, data_dir: str, n_mels: int, target_length: int, n_fft: int, cache_dir: str,
        device: torch.device, suppress_warnings: bool = False,
        suppress_reading_messages: bool = False, num_workers: int = 1, stem_names: List[str] = None,
        stop_flag: Any = None, device_prep: torch.device = None, segments_per_track: int = 10,
        file_ids: List[Dict[str, Any]] = None
    ):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.device = device
        self.suppress_warnings = suppress_warnings
        self.suppress_reading_messages = suppress_reading_messages
        self.num_workers = num_workers
        self.stop_flag = stop_flag
        self.device_prep = device_prep or device
        self.segments_per_track = segments_per_track

        self.sample_rate, _, _ = self._detect_parameters()
        self.stem_names = stem_names or self._infer_stem_names()
        self.file_ids = file_ids if file_ids is not None else self._get_file_ids()
        os.makedirs(cache_dir, exist_ok=True)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=n_fft, n_mels=n_mels,
            win_length=None, hop_length=n_fft // 4, power=2.0
        ).to(self.device_prep)
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device_prep)

    def _detect_parameters(self):
        try:
            return detect_parameters_from_cache(self.cache_dir)
        except ValueError:
            return detect_parameters_from_raw_data(self.data_dir)

    def _infer_stem_names(self) -> List[str]:
        stem_files = os.listdir(self.data_dir)
        stems = set()
        for file in stem_files:
            if file.startswith('example_') and file.endswith('.ogg'):
                continue
            stem = file.split('_')[0]
            stems.add(stem)
        return list(stems)

    def _get_file_ids(self):
        file_ids = []
        for input_file in os.listdir(self.data_dir):
            if input_file.startswith('example_') and input_file.endswith('.ogg'):
                identifier = input_file.split('_')[1].split('.')[0]
                target_files = {
                    stem: f"{stem}_{identifier}.ogg" for stem in self.stem_names
                }
                file_ids.append({
                    'identifier': identifier,
                    'input_file': input_file,
                    'target_files': target_files
                })
        return file_ids

    def _get_cache_path(self, stem_name: str, identifier: str) -> str:
        return os.path.join(self.cache_dir, f"{stem_name}_{identifier}_{self.n_mels}_{self.target_length}_{self.n_fft}.h5")

    def process_and_cache_file(self, file_path: str, identifier: str, stem_name: str) -> torch.Tensor:
        cache_path = self._get_cache_path(stem_name, identifier)

        if os.path.exists(cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            with h5py.File(cache_path, 'r') as f:
                return torch.from_numpy(f['audio'][:]).to(self.device)

        logger.info(f"Processing file: {file_path}")
        data, sr = sf.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

        segments = segment_audio(data, chunk_size=self.target_length)
        processed_segments = []

        for i, segment in enumerate(segments):
            segment_tensor = torch.tensor(segment).float().to(self.device).unsqueeze(0).unsqueeze(0)
            mel_spec = self.mel_spectrogram(segment_tensor)
            mel_spec = self.amplitude_to_db(mel_spec)

            if mel_spec.size(2) == 0:
                logger.warning(f"Skipping segment {i} for {file_path} due to zero-sized mel spectrogram.")
                continue

            harmonic, percussive = librosa.effects.hpss(segment)
            harmonic_tensor = torch.tensor(harmonic).float().to(self.device).unsqueeze(0).unsqueeze(0)
            percussive_tensor = torch.tensor(percussive).float().to(self.device).unsqueeze(0).unsqueeze(0)

            harmonic_spec = self.mel_spectrogram(harmonic_tensor)
            percussive_spec = self.mel_spectrogram(percussive_tensor)

            harmonic_spec = self.amplitude_to_db(harmonic_spec)
            percussive_spec = self.amplitude_to_db(percussive_spec)

            combined_spec = torch.cat([mel_spec, harmonic_spec, percussive_spec], dim=1)
            combined_spec = F.pad(combined_spec, (0, self.target_length - combined_spec.size(-1)))
            processed_segments.append(combined_spec)

        if len(processed_segments) == 0:
            logger.warning(f"No valid segments found for processing in {file_path}. Skipping file.")
            return torch.tensor([]).to(self.device)

        combined_spec = torch.stack(processed_segments)
        logger.info(f"Final combined_spec size for {file_path}: {combined_spec.size()}")

        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('audio', data=combined_spec.cpu().numpy(), compression="gzip")
            f.attrs['sample_rate'] = self.sample_rate

        logger.info(f"Successfully processed and cached file: {file_path}")
        return combined_spec

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.file_ids[idx]
        identifier = file_id['identifier']

        input_cache = self._get_cache_path('input', identifier)
        input_data = self.process_and_cache_file(file_id['input_file'], identifier, 'input')

        target_data = {}
        for stem in self.stem_names:
            target_cache = self._get_cache_path(stem, identifier)
            target_data[stem] = self.process_and_cache_file(file_id['target_files'][stem], identifier, stem)

        return {"input": input_data, "target": target_data, "file_id": identifier}

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[str]]]:
    inputs = torch.cat([item['input'] for item in batch if item['input'].numel() > 0], dim=0)

    targets = {stem: [] for stem in batch[0]['target'].keys()}
    for item in batch:
        for stem, target in item['target'].items():
            if target.numel() > 0:
                targets[stem].append(target)

    for stem in targets:
        if targets[stem]:
            targets[stem] = torch.cat(targets[stem], dim=0)
        else:
            targets[stem] = torch.tensor([])  # Ensuring it's always a tensor

    file_ids = [item['file_id'] for item in batch]
    return {'input': inputs, 'target': targets, 'file_paths': file_ids}

def create_dataloader(dataset: StemSeparationDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def log_tensor_dimensions(tensor: torch.Tensor, message: str):
    logger.info(f"{message} - Tensor shape: {tensor.shape}")

def process_and_cache_dataset(
    data_dir: str, cache_dir: str, n_mels: int, n_fft: int,
    device: torch.device, suppress_reading_messages: bool
):
    dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=22050,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        segments_per_track=10
    )

    for i in range(len(dataset)):
        file_id = dataset.file_ids[i]
        if file_id is not None:
            identifier = file_id['identifier']
            input_cache_file_path = dataset._get_cache_path('input', identifier)
            target_cache_file_paths = [dataset._get_cache_path(stem, identifier) for stem in dataset.stem_names]

            if os.path.exists(input_cache_file_path) and all(os.path.exists(p) for p in target_cache_file_paths):
                logger.info(f"Skipping processing for {identifier} as cache files already exist.")
                continue

            input_file_path = os.path.join(data_dir, file_id['input_file'])
            dataset.process_and_cache_file(input_file_path, identifier, 'input')

            for stem_name in dataset.stem_names:
                target_file_path = os.path.join(data_dir, file_id['target_files'][stem_name])
                dataset.process_and_cache_file(target_file_path, identifier, stem_name)

            logger.info(f"Processed and cached file {i+1}/{len(dataset)}")

    logger.info("Dataset preprocessing and caching completed.")

    num_files = len(dataset.file_ids)
    split_index = int(num_files * 0.8)
    train_file_ids = dataset.file_ids[:split_index]
    val_file_ids = dataset.file_ids[split_index:]
    return train_file_ids, val_file_ids

def integrated_dynamic_batching(data_dir: str, cache_dir: str, n_mels: int, n_fft: int, target_length: int, batch_size: int, device: torch.device, stem_name: str, file_ids: List[Dict[str, Any]], shuffle: bool = False):
    dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=target_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        device=device,
        file_ids=file_ids
    )
    
    if shuffle:
        random.shuffle(file_ids)

    for i in range(0, len(file_ids), batch_size):
        batch_file_ids = file_ids[i:i + batch_size]

        warmed_inputs = []
        warmed_targets = []
        warmed_file_paths = []

        for file_id in batch_file_ids:
            identifier = file_id['identifier']
            input_cache_file_name = f"input_{identifier}_{n_mels}_{target_length}_{n_fft}.h5"
            target_cache_file_name = f"{stem_name}_{identifier}_{n_mels}_{target_length}_{n_fft}.h5"
            input_cache_file_path = os.path.join(cache_dir, input_cache_file_name)
            target_cache_file_path = os.path.join(cache_dir, target_cache_file_name)

            if os.path.exists(input_cache_file_path) and os.path.exists(target_cache_file_path):
                logger.info(f"Cache files found for {identifier}. Skipping processing.")
            else:
                logger.warning(f"Cache files not found for {identifier}. Processing now.")
                input_file_path = os.path.join(data_dir, f'example_{identifier}.ogg')
                dataset.process_and_cache_file(input_file_path, identifier, "input")

                target_file_path = os.path.join(data_dir, f'{stem_name}_{identifier}.ogg')
                dataset.process_and_cache_file(target_file_path, identifier, stem_name)

            data = load_from_cache(input_cache_file_path, device)
            warmed_inputs.append(data['input'])
            warmed_targets.append(load_from_cache(target_cache_file_path, device)['input'])
            warmed_file_paths.append(identifier)

        yield {
            'input': torch.stack(warmed_inputs),
            'target': {stem_name: torch.stack(warmed_targets)},
            'file_paths': warmed_file_paths
        }

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

def load_from_cache(cache_file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        with h5py.File(cache_file_path, 'r') as f:
            keys = list(f.keys())
            logger.info(f"Keys in cache file {cache_file_path}: {keys}")
            input_data = torch.tensor(f['audio'][:], device=device).float()

        logger.info(f"Loaded input data shape: {input_data.shape}")

        return {'input': input_data}
    except Exception as e:
        logger.error(f"Error loading from cache file '{cache_file_path}': {e}")
        raise

def ensure_dir_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# Ensure the logging configuration is properly set up
setup_logger(__name__)
