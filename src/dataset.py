import os
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.multiprocessing import Queue, Process, Value, Lock
import logging
import queue
import hashlib
from utils import load_and_preprocess
from worker import worker

logger = logging.getLogger(__name__)

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=False, num_workers=4, device='cpu'):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_messages = suppress_messages
        self.num_workers = num_workers
        self.device = device
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.valid_stems = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.ogg'))]

        self.num_processed_stems = Value('i', 0)
        self.lock = Lock()

        self.mel_spectrogram_params = {
            'sample_rate': 22050,
            'n_fft': self.n_fft,
            'win_length': None,
            'hop_length': self.n_fft // 4,
            'n_mels': self.n_mels,
            'power': 2.0
        }

        self.workers = self._start_workers()
        self.processed_data = []
        self._process_stems()

    def _get_cache_key(self, stem_name):
        cache_key = f"{stem_name}_{self.apply_data_augmentation}_{self.mel_spectrogram_params}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        logger.debug(f"Generated cache key for {stem_name}: {cache_key}")
        return cache_key

    def _get_cache_path(self, cache_key):
        cache_subdir = "augmented" if self.apply_data_augmentation else "original"
        cache_dir = os.path.join(self.cache_dir, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_key + ".pt")

    def _load_from_cache(self, cache_path):
        try:
            logger.info(f"Attempting to load from cache: {cache_path}")
            data = torch.load(cache_path)
            logger.info(f"Successfully loaded from cache: {cache_path}")
            return data
        except (FileNotFoundError, RuntimeError):  # Handle missing or corrupted cache
            logger.warning(f"Cache not found or corrupted: {cache_path}")
            return None

    def _save_to_cache(self, cache_path, data):
        logger.info(f"Saving to cache: {cache_path}")
        torch.save(data, cache_path)

    def _start_workers(self):
        workers = []
        stems_per_worker = len(self.valid_stems) // self.num_workers
        for i in range(self.num_workers):
            start_idx = i * stems_per_worker
            end_idx = start_idx + stems_per_worker if i < self.num_workers - 1 else len(self.valid_stems)
            worker_stems = self.valid_stems[start_idx:end_idx]
            
            p = Process(target=worker, args=(
                self.input_queue, self.output_queue, self.mel_spectrogram_params, 
                self.data_dir, self.target_length, str(self.device), self.apply_data_augmentation,
                worker_stems, self.lock, self.num_processed_stems, self.cache_dir
            ))
            p.start()
            workers.append(p)
        return workers

    def _process_stems(self):
        processed_data = []
        while True:
            try:
                stem_name, data = self.output_queue.get(timeout=60)
                if data is None:
                    continue
                logger.info(f"Processed stem: {stem_name}")
                processed_data.append((stem_name, data))
            except queue.Empty:
                with self.lock:
                    if self.num_processed_stems.value >= len(self):
                        logger.info("All stems processed, ending iteration.")
                        break
            except (ConnectionResetError, EOFError) as e:
                logger.error(f"Connection error in _process_stems: {e}")
                self._restart_workers()

        processed_data.sort(key=lambda x: x[0])
        self.processed_data = [x[1] for x in processed_data]

    def _restart_workers(self):
        logger.info("Restarting workers due to connection error.")
        for worker in self.workers:
            worker.terminate()
        self.workers = self._start_workers()

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
        self._save_to_cache(cache_path, data)
        return data

    def _process_single_stem(self, stem_name):
        file_path = os.path.join(self.data_dir, stem_name)
        logger.info(f"Loading and preprocessing: {file_path}")
        mel_spectrogram = T.MelSpectrogram(**self.mel_spectrogram_params).to(self.device)
        return load_and_preprocess(file_path, mel_spectrogram, self.target_length, self.apply_data_augmentation, self.device)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
