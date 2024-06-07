import os
import torch
import torchaudio.transforms as T
import logging
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
from preprocessing_utils import load_and_preprocess, load_from_cache

logger = logging.getLogger(__name__)

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

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = self._load_cache_metadata()
        else:
            self.cache = {}

    def _load_cache_metadata(self):
        cache_path = os.path.join(self.cache_dir, "cache_metadata.pt")
        if os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path)
                if not self.suppress_reading_messages:
                    logger.info(f"Loaded cache metadata from {cache_path}")
                return cache
            except Exception as e:
                if not self.suppress_reading_messages:
                    logger.error(f"Error loading cache metadata file: {e}")
                return {}
        return {}

    def _save_cache_metadata(self):
        cache_path = os.path.join(self.cache_dir, "cache_metadata.pt")
        try:
            if not self.suppress_reading_messages:
                logger.info(f"Saving cache metadata to {cache_path}")
            torch.save(self.cache, cache_path)
            if not self.suppress_reading_messages:
                logger.info(f"Successfully saved cache metadata to {cache_path}")
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error saving cache metadata file: {e}")

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        if self.stop_flag.value == 1:
            return None

        stem_name = self.valid_stems[idx]
        cache_key = self._get_cache_key(stem_name)

        if self.use_cache and cache_key in self.cache:
            stem_cache_path = self.cache[cache_key]
            if os.path.exists(stem_cache_path):
                try:
                    data = load_from_cache(stem_cache_path)
                    data = {'input': data, 'target': data.clone()}  # Assuming input and target are the same in this example
                    if self._validate_data(data):
                        if not self.suppress_reading_messages:
                            logger.info(f"Loaded cached data for {stem_name}")
                        data['input'] = data['input'].to(self.device_prep)
                        data['target'] = data['target'].to(self.device_prep)
                        return data
                    else:
                        raise ValueError(f"Invalid cached data for {stem_name}.")
                except Exception as e:
                    raise ValueError(f"Error loading cached data for {stem_name}. Error: {e}")
            else:
                raise ValueError(f"Cache file not found for {stem_name}.")
        else:
            raise ValueError("Cache not found and use_cache is True.")

    def _get_cache_key(self, stem_name):
        return f"{stem_name}_{self.apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"

    def _save_individual_cache(self, stem_name, data):
        stem_cache_path = os.path.join(self.cache_dir, f"{stem_name}.h5")
        try:
            if not self.suppress_reading_messages:
                logger.info(f"Saving stem cache: {stem_cache_path}")
            with h5py.File(stem_cache_path, 'w') as f:
                f.create_dataset('input', data=data['input'].cpu().numpy())
                f.create_dataset('target', data=data['target'].cpu().numpy())
            if not self.suppress_reading_messages:
                logger.info(f"Successfully saved stem cache: {stem_cache_path}")
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

        input_shape = data['input'].shape
        target_shape = data['target'].shape
        expected_shape = (1, self.n_mels, self.target_length)

        if input_shape != expected_shape or target_shape != expected_shape:
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: shape mismatch. Input shape: {input_shape}, Target shape: {target_shape}, Expected shape: {expected_shape}")
            return False

        if not isinstance(data['input'], torch.Tensor) or not isinstance(data['target'], torch.Tensor):
            if not self.suppress_reading_messages:
                logger.warning(f"Data validation failed: 'input' or 'target' is not a torch.Tensor. Input type: {type(data['input'])}, Target type: {type(data['target'])}")
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

        cache_key = self._get_cache_key(stem_name)

        if cache_key in self.cache:
            stem_cache_path = self.cache[cache_key]
            if os.path.exists(stem_cache_path):
                try:
                    data = load_from_cache(stem_cache_path)
                    data = {'input': data, 'target': data.clone()}  # Assuming input and target are the same in this example
                    if self._validate_data(data):
                        return data
                    else:
                        if not self.suppress_reading_messages:
                            logger.warning(f"Invalid cached data for {stem_name}. Reprocessing.")
                        os.remove(stem_cache_path)
                except Exception as e:
                    if not self.suppress_reading_messages:
                        logger.warning(f"Error loading cached data for {stem_name}. Reprocessing. Error: {e}")
                    os.remove(stem_cache_path)

        raise ValueError(f"Cache file not found for {stem_name}.")

def pad_tensor(tensor, target_length, target_width):
    current_length = tensor.size(2)
    current_width = tensor.size(1)
    if current_length < target_length or current_width < target_width:
        padding = (0, target_length - current_length, 0, target_width - current_width)
        tensor = torch.nn.functional.pad(tensor, padding)
    return tensor

def collate_fn(batch):
    max_length = max(item['input'].size(2) for item in batch)
    max_width = max(item['input'].size(1) for item in batch)

    inputs = torch.stack([pad_tensor(item['input'], max_length, max_width) for item in batch])
    targets = torch.stack([pad_tensor(item['target'], max_length, max_width) for item in batch])
    
    return {'input': inputs, 'target': targets}
