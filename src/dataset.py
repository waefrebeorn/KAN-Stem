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
        self.cache = self._load_cache_metadata()

    def _load_cache_metadata(self):
        cache_path = os.path.join(self.cache_dir, "cache_metadata.pt")
        if os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path)
                logger.info(f"Loaded cache metadata from {cache_path}")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache metadata file: {e}")
                return {}
        return {}

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        stem_name = self.valid_stems[idx]
        cache_key = self._get_cache_key(stem_name)

        if cache_key in self.cache:
            stem_cache_path = self.cache[cache_key]
            if os.path.exists(stem_cache_path):
                try:
                    data = torch.load(stem_cache_path)
                    if data and 'input' in data and 'target' in data:
                        return data
                    else:
                        logger.warning(f"Incomplete data in cache for {stem_name}. Reprocessing.")
                        os.remove(stem_cache_path)
                except Exception as e:
                    logger.error(f"Error loading cached data for {stem_name}: {e}")

        logger.info(f"Processing stem: {stem_name}")
        data = self._process_single_stem(stem_name)
        if data is not None:
            data['input'] = data['input'].cpu()
            data['target'] = data['target'].cpu()
            self.cache[cache_key] = self._save_individual_cache(stem_name, data)
            self._save_cache_metadata()
        return data

    def _get_cache_key(self, stem_name):
        return f"{stem_name}_{self.apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"

    def _save_individual_cache(self, stem_name, data):
        stem_cache_path = os.path.join(self.cache_dir, f"{stem_name}.pt")
        try:
            logger.info(f"Saving stem cache: {stem_cache_path}")
            torch.save(data, stem_cache_path)
            logger.info(f"Successfully saved stem cache: {stem_cache_path}")
            return stem_cache_path
        except Exception as e:
            logger.error(f"Error saving stem cache: {stem_cache_path}. Error: {e}")
            return None

    def _save_cache_metadata(self):
        cache_path = os.path.join(self.cache_dir, "cache_metadata.pt")
        try:
            logger.info(f"Saving cache metadata to {cache_path}")
            torch.save(self.cache, cache_path)
            logger.info(f"Successfully saved cache metadata to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache metadata file: {e}")

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

            # Ensure the mel spectrogram has the expected shape
            if input_mel.shape[-1] < self.target_length:
                input_mel = torch.nn.functional.pad(input_mel, (0, self.target_length - input_mel.shape[-1]), mode='constant')

            target_mel = input_mel.clone()
            input_mel = input_mel.unsqueeze(0)
            target_mel = target_mel.unsqueeze(0)

            # Debugging statements
            logger.debug(f"Processed input mel shape: {input_mel.shape}")
            logger.debug(f"Processed target mel shape: {target_mel.shape}")

            return {"input": input_mel, "target": target_mel}
        except Exception as e:
            logger.error(f"Error processing stem {stem_name}: {e}")
            return None

def collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {'input': inputs, 'target': targets}
