import os
import torch
import torchaudio.transforms as T
from torch.multiprocessing import Queue, current_process
import logging
import time
from utils import load_and_preprocess
import hashlib
import pickle

logger = logging.getLogger(__name__)

def worker(input_queue, output_queue, mel_spectrogram_params, data_dir, target_length, device_str, apply_data_augmentation, valid_stems, lock, num_processed_stems, cache_dir):
    device = torch.device(device_str)
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params).to(device)
    process_name = f"Worker-{current_process().pid}"
    logger.info(f"{process_name}: Started with {len(valid_stems)} stems to process.")

    def get_cache_key(stem_name):
        cache_key = f"{stem_name}_{apply_data_augmentation}_{mel_spectrogram_params}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        logger.debug(f"{process_name}: Generated cache key for {stem_name}: {cache_key}")
        return cache_key

    def get_cache_path(cache_key):
        cache_subdir = "augmented" if apply_data_augmentation else "original"
        cache_dir_path = os.path.join(cache_dir, cache_subdir)
        os.makedirs(cache_dir_path, exist_ok=True)
        return os.path.join(cache_dir_path, cache_key + ".pt")

    for stem_name in valid_stems:
        try:
            file_path = os.path.join(data_dir, stem_name)
            logger.info(f"{process_name}: Processing file: {file_path}")

            cache_key = get_cache_key(stem_name)
            cache_path = get_cache_path(cache_key)
            if os.path.exists(cache_path):
                logger.info(f"{process_name}: Loading from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"{process_name}: Successfully loaded from cache: {cache_path}")
            else:
                start_time = time.time()
                data = load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device)
                end_time = time.time()

                if data is not None:
                    logger.info(f"{process_name}: Successfully processed {file_path} in {end_time - start_time:.2f} seconds")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"{process_name}: Saved to cache: {cache_path}")
                else:
                    logger.warning(f"{process_name}: Failed to process {file_path}")

            output_queue.put((stem_name, data))

            with lock:
                num_processed_stems.value += 1

        except Exception as e:
            logger.error(f"{process_name}: Error processing {file_path}: {e}")
            output_queue.put((stem_name, None))

    output_queue.close()
    logger.info(f"{process_name}: Finished processing. Output queue closed.")
