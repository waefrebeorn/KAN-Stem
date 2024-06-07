import os
import torch
import torchaudio.transforms as T
import logging
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocessing_utils import load_and_preprocess  # Import from the new location

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
        if self.cache != self._load_cache_metadata():
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
                    data = torch.load(stem_cache_path)
                    if self._validate_data(data):
                        if not self.suppress_reading_messages:
                            logger.info(f"Loaded cached data for {stem_name}")
                        return data
                    else:
                        if not self.suppress_reading_messages:
                            logger.warning(f"Invalid cached data for {stem_name}. Reprocessing.")
                        os.remove(stem_cache_path)
                except Exception as e:
                    if not self.suppress_reading_messages:
                        logger.warning(f"Error loading cached data for {stem_name}. Reprocessing. Error: {e}")
                    os.remove(stem_cache_path)

        if not self.suppress_reading_messages:
            logger.info(f"Processing stem: {stem_name}")
        data = self._process_single_stem(stem_name)
        if data is not None and self.use_cache:
            data['input'] = data['input'].cpu()
            data['target'] = data['target'].cpu()
            self.cache[cache_key] = self._save_individual_cache(stem_name, data)
            self._save_cache_metadata()
        return data

    def _get_cache_key(self, stem_name):
        return f"{stem_name}_{self.apply_data_augmentation}_{self.n_mels}_{self.target_length}_{self.n_fft}"

    def _save_individual_cache(self, stem_name, data):
        stem_cache_path = os.path.join(self.cache_dir, f"{stem_name}.pt")
        if not os.path.exists(stem_cache_path):
            try:
                if not self.suppress_reading_messages:
                    logger.info(f"Saving stem cache: {stem_cache_path}")
                torch.save(data, stem_cache_path)
                if not self.suppress_reading_messages:
                    logger.info(f"Successfully saved stem cache: {stem_cache_path}")
                return stem_cache_path
            except Exception as e:
                if not self.suppress_reading_messages:
                    logger.error(f"Error saving stem cache: {stem_cache_path}. Error: {e}")
                return None
        else:
            return stem_cache_path

    def _process_single_stem(self, stem_name):
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

            input_mel = load_and_preprocess(file_path, mel_spectrogram, self.target_length, self.apply_data_augmentation, self.device_prep)

            if input_mel is None:
                return None

            if input_mel.shape[-1] < self.target_length:
                input_mel = torch.nn.functional.pad(input_mel, (0, self.target_length - input_mel.shape[-1]), mode='constant')

            target_mel = input_mel.clone()
            input_mel = input_mel.unsqueeze(0)
            target_mel = target_mel.unsqueeze(0)

            logger.debug(f"Processed input mel shape: {input_mel.shape}")
            logger.debug(f"Processed target mel shape: {target_mel.shape}")

            return {"input": input_mel, "target": target_mel}
        except Exception as e:
            if not self.suppress_reading_messages:
                logger.error(f"Error processing stem {stem_name}: {e}")
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
                    data = torch.load(stem_cache_path)
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

        data = self._process_single_stem(stem_name)
        if data is not None and self.use_cache:
            data['input'] = data['input'].cpu()
            data['target'] = data['target'].cpu()
            self.cache[cache_key] = self._save_individual_cache(stem_name, data)
            self._save_cache_metadata()
        return data

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

class OnTheFlyPreprocessingDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, apply_data_augmentation, device_prep):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.apply_data_augmentation = apply_data_augmentation
        self.device_prep = device_prep

        self.valid_stems = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        if not self.valid_stems:
            raise ValueError(f"No valid audio files found in {data_dir}")

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        stem_name = self.valid_stems[idx]

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

        if input_mel.shape[-1] < self.target_length:
            input_mel = torch.nn.functional.pad(input_mel, (0, self.target_length - input_mel.shape[-1]), mode='constant')

        target_mel = input_mel.clone()
        input_mel = input_mel.unsqueeze(0)
        target_mel = target_mel.unsqueeze(0)

        logger.debug(f"Processed input mel shape: {input_mel.shape}")
        logger.debug(f"Processed target mel shape: {target_mel.shape}")

        return {"input": input_mel, "target": target_mel}
