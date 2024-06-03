import logging
from dataset import StemSeparationDataset
from utils import detect_parameters

logger = logging.getLogger(__name__)

def preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep):
    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep)
    
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
        except Exception as e:
            logger.error(f"Error during preprocessing and caching: {e}")
    
    logger.info("Preprocessing and caching completed.")
