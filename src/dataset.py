import os
import torch
import torchaudio.transforms as T
import logging
from torch.utils.data import Dataset
from utils import load_and_preprocess

logger = logging.getLogger(__name__)

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_warnings = suppress_warnings
        self.num_workers = num_workers
        self.device_prep = device_prep

        self.valid_stems = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        if not self.valid_stems:
            raise ValueError(f"No valid audio files found in {data_dir}")

        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        stem_name = self.valid_stems[idx]
        cache_key = self._get_cache_key(stem_name)
        cache_path = self._get_cache_path(cache_key)

        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data

        logger.info(f"Processing stem: {stem_name}")
        data = self._process_single_stem(stem_name)
        if data is not None:
            # Move tensors to CPU before saving to cache
            data['input'] = data['input'].cpu()
            data['target'] = data['target'].cpu()
            self._save_to_cache(cache_path, data)
        return data

    def _get_cache_key(self, stem_name):
        return f"{stem_name}_{self.apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"

    def _get_cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.pt")

    def _load_from_cache(self, cache_path):
        try:
            logger.info(f"Attempting to load from cache: {cache_path}")
            data = torch.load(cache_path)
            logger.info(f"Successfully loaded from cache: {cache_path}")
            return data
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Cache not found or corrupted: {cache_path}. Error: {e}")
            return None

    def _save_to_cache(self, cache_path, data):
        try:
            logger.info(f"Saving to cache: {cache_path}")
            torch.save(data, cache_path)
            logger.info(f"Successfully saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving to cache: {cache_path}. Error: {e}")

    def _process_single_stem(self, stem_name):
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

            input_mel = load_and_preprocess(file_path, mel_spectrogram, self.target_length, self.apply_data_augmentation, self.device_prep)

            if input_mel is None:
                return None

            # Simulate target data for now. In practice, this should be the actual target stem
            target_mel = input_mel.clone()

            # Add an extra dimension to make it 4D: (batch_size, channels, height, width)
            input_mel = input_mel.unsqueeze(0)
            target_mel = target_mel.unsqueeze(0)

            return {"input": input_mel, "target": target_mel}
        except Exception as e:
            logger.error(f"Error processing stem {stem_name}: {e}")
            return None

def collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {'input': inputs, 'target': targets}
