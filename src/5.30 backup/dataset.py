import os
import torch
from torch.utils.data import Dataset
from multiprocessing import Queue, Process, current_process
import torchaudio.transforms as T
import logging
import queue
import time
from utils import logger, read_audio, data_augmentation, load_and_preprocess

def worker(input_queue, output_queue, mel_spectrogram_params, data_dir, target_length, apply_data_augmentation, valid_stems):
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params)
    process_name = f"Worker-{current_process().pid}"
    logger.info(f"{process_name}: Started with {len(valid_stems)} stems to process.")

    for stem_name in valid_stems:
        try:
            file_path = os.path.join(data_dir, stem_name)
            logger.info(f"{process_name}: Processing file: {file_path}")

            start_time = time.time()
            data = load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation)
            end_time = time.time()

            if data is not None:
                logger.info(f"{process_name}: Successfully processed {file_path} in {end_time - start_time:.2f} seconds")
                output_queue.put((stem_name, data))  # Put the stem name and the processed data into the output queue
            else:
                logger.warning(f"{process_name}: Failed to process {file_path}")
                output_queue.put((stem_name, None))

        except Exception as e:
            logger.error(f"{process_name}: Error processing {file_path}: {e}")
            output_queue.put((stem_name, None))

    output_queue.close()
    logger.info(f"{process_name}: Finished processing. Output queue closed.")

def worker_init_fn(worker_id):
    torch.manual_seed(torch.initial_seed() + worker_id)

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=False, num_workers=4):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_messages = suppress_messages
        self.mel_spectrogram_params = {'n_mels': n_mels, 'n_fft': n_fft}
        self.valid_stems = self._get_valid_stems()
        self.num_workers = num_workers

        if suppress_messages:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logger.info(f"Initializing dataset with {len(self.valid_stems)} valid stems")

        self.input_queue = Queue()
        self.output_queue = Queue(maxsize=100)
        self.workers = self._start_workers()

    def _get_valid_stems(self):
        stems = [f for f in os.listdir(self.data_dir) if f.startswith("input") and f.endswith(".wav")]
        logger.info(f"Valid stems found: {stems}")
        return stems

    def _start_workers(self):
        workers = []
        stems_per_worker = len(self.valid_stems) // self.num_workers
        for i in range(self.num_workers):
            start_idx = i * stems_per_worker
            end_idx = start_idx + stems_per_worker if i < self.num_workers - 1 else len(self.valid_stems)
            worker_stems = self.valid_stems[start_idx:end_idx]
            p = Process(target=worker, args=(self.input_queue, self.output_queue, self.mel_spectrogram_params, self.data_dir, self.target_length, self.apply_data_augmentation, worker_stems))
            p.start()
            workers.append(p)
        return workers

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, index):
        stem_name = self.valid_stems[index]
        logger.debug(f"Putting index {index} and stem_name {stem_name} into input queue")
        self.input_queue.put((index, stem_name))

        while True:
            try:
                stem_name, data = self.output_queue.get()
                if data is not None:
                    return data
                logger.warning(f"Received None for stem_name {stem_name}. Retrying.")
                self.input_queue.put((index, stem_name))
            except queue.Empty:
                pass
            except (ConnectionResetError, EOFError) as e:
                logger.error(f"Connection error in __getitem__: {e}")
                self._restart_workers()
                self.input_queue.put((index, stem_name))

        return data

    def _restart_workers(self):
        logger.warning("Restarting workers...")
        for worker in self.workers:
            worker.terminate()
        
        self.input_queue = Queue()
        self.output_queue = Queue(maxsize=100)
        self.workers = self._start_workers()

    def __del__(self):
        for _ in self.workers:
            self.input_queue.put(None)

        for p in self.workers:
            p.join()

        self.input_queue.close()
        self.output_queue.close()

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
