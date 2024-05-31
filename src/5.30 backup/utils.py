import os
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import soundfile as sf
from torchaudio import transforms as T
import psutil
import GPUtil
import time
from multiprocessing import Queue, Process, Value, Lock, current_process
from torch.utils.data import Dataset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress debug messages from httpcore and other related libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global variables to track the number of processed stems
num_processed_stems = Value('i', 0)
total_num_stems = Value('i', 0)  # Total number of stems
dataset_lock = Lock()

# Helper function to analyze audio file
def analyze_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / sample_rate
        is_silent = waveform.abs().max() == 0
        return sample_rate, duration, is_silent
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        return None, None, True

# Helper function to detect parameters for the dataset
def detect_parameters(data_dir, default_n_mels=64, default_n_fft=1024):
    sample_rates = []
    durations = []

    logger.info(f"Contents of the data directory ({data_dir}): {os.listdir(data_dir)}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            sample_rate, duration, is_silent = analyze_audio(file_path)
            if is_silent:
                logger.info(f"Skipping silent file: {file_name}")
                continue
            sample_rates.append(sample_rate)
            durations.append(duration)

    logger.info(f"Found {len(sample_rates)} valid audio files")

    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    avg_sample_rate = sum(sample_rates) / len(sample_rates)
    avg_duration = sum(durations) / len(durations)

    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))

    return int(avg_sample_rate), n_mels, n_fft

# Multi-Scale Spectral Loss
class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, n_ffts=[2048, 1024, 512, 256, 128]):
        super(MultiScaleSpectralLoss, self).__init__()
        self.n_ffts = n_ffts

    def forward(self, y_true, y_pred):
        loss = 0.0
        for n_fft in self.n_ffts:
            y_true_stft = torch.stft(y_true, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, return_complex=True)
            y_pred_stft = torch.stft(y_pred, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, return_complex=True)
            loss += F.l1_loss(torch.abs(y_true_stft), torch.abs(y_pred_stft))
        return loss

# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * len(self.feature_extractor)

    def forward(self, y_true, y_pred):
        loss = 0.0
        features_true = self.feature_extractor(y_true)
        features_pred = self.feature_extractor(y_pred)
        for weight, feature_true, feature_pred in zip(self.layer_weights, features_true, features_pred):
            loss += weight * F.l1_loss(feature_true, feature_pred)
        return loss

# Compute adversarial loss
def compute_adversarial_loss(discriminator, y_true, y_pred, device):
    real_labels = torch.ones(y_true.size(0), 1, device=device)
    fake_labels = torch.zeros(y_pred.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy_with_logits(discriminator(y_true), real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(y_pred), fake_labels)
    return (real_loss + fake_loss) / 2

# Read audio
def read_audio(file_path, device='cuda' if torch.cuda.is_available() else 'cpu', suppress_messages=False):
    try:
        if not suppress_messages:
            logger.info(f"Attempting to read: {file_path}")
        data, samplerate = torchaudio.load(file_path)
        data = data.to(device)
        return data, samplerate
    except FileNotFoundError:
        logger.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
    return None, None

# Data augmentation
def data_augmentation(inputs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2).to(device)
        freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
        time_mask = T.TimeMasking(time_mask_param=35).to(device)
        
        augmented_inputs = pitch_shift(inputs.clone().detach().to(device))
        augmented_inputs = freq_mask(augmented_inputs.clone().detach())
        augmented_inputs = time_mask(augmented_inputs.clone().detach())
        
        return augmented_inputs
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        return inputs

# Get optimizer
def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Momentum":
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSProp":
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, alpha=0.99, weight_decay=weight_decay)
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# Custom loss
def custom_loss(output, target, loss_fn=nn.SmoothL1Loss()):
    try:
        if output.shape != target.shape:
            min_length = min(output.size(-1), target.size(-1))
            output = output[..., :min_length]
            target = target[..., :min_length]
        loss = loss_fn(output, target)
        return loss
    except Exception as e:
        logger.error(f"Error in custom loss calculation: {e}")
        return None

# Write audio
def write_audio(file_path, data, samplerate):
    try:
        data_cpu = data.squeeze(0).cpu().numpy()
        sf.write(file_path, data_cpu, samplerate)
    except Exception as e:
        logger.error(f"Error writing audio file {file_path}: {e}")

# Check device
def check_device(model, *args):
    try:
        device = next(model.parameters()).device
        for idx, arg in enumerate(args):
            if arg.device != device:
                raise RuntimeError(
                    f"Argument {idx} is on {arg.device}, but expected it to be on {device}"
                )
    except Exception as e:
        logger.error(f"Error in device check: {e}")

# Load and preprocess function
def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.debug("Starting load and preprocess")
        input_audio, _ = read_audio(file_path, device='cpu')
        if input_audio is None:
            return None

        logger.debug(f"Device for processing: {device}")

        # Ensure the mel_spectrogram and all related operations use the same device
        mel_spectrogram = mel_spectrogram.to(device)
        input_audio = input_audio.float().to(device)
        logger.debug(f"input_audio device: {input_audio.device}")

        if apply_data_augmentation:
            input_audio = data_augmentation(input_audio, device=device)
            logger.debug(f"After data augmentation, input_audio device: {input_audio.device}")

        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]
        logger.debug(f"input_mel device: {input_mel.device}")

        logger.debug("Completed load and preprocess")
        return input_mel

    except Exception as e:
        logger.error(f"Error in load_and_preprocess: {e}")
        return None

# Monitor system resources
def log_system_resources():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None

        logger.info(f"CPU usage: {cpu_percent}%")
        logger.info(f"Memory usage: {memory_info.percent}%")
        if gpu_info:
            logger.info(f"GPU usage: {gpu_info.load * 100:.2f}%")
            logger.info(f"GPU memory usage: {gpu_info.memoryUtil * 100:.2f}%")
    except Exception as e:
        logger.error(f"Error logging system resources: {e}")

# Example function to call for each worker
def process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.info(f"Processing file: {file_path}")
        log_system_resources()
        start_time = time.time()

        result = load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device)
        if result is not None:
            # Do something with the result, e.g., save to disk or further processing
            pass

        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")

        # Additional debug logging after processing
        logger.debug(f"Post-processing check for file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")

# Example usage in a worker loop
def worker_loop(file_paths, mel_spectrogram_params, target_length, apply_data_augmentation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params).to(device)

    try:
        for file_path in file_paths:
            logger.debug(f"Starting processing for file: {file_path}")
            process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device)
            logger.debug(f"Finished processing for file: {file_path}")
        logger.debug("All files processed in worker loop")

        # Additional debug logging after all files are processed
        logger.debug("Entering post-processing stage after worker loop")
        # Any additional post-processing can be added here
        logger.debug("Completed post-processing stage")

    except Exception as e:
        logger.error(f"Error in worker loop: {e}")

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, mel_spectrogram_params, target_length, apply_data_augmentation, num_workers=4):
        self.data_dir = data_dir
        self.mel_spectrogram_params = mel_spectrogram_params
        self.target_length = target_length
        self.apply_data_augmentation = apply_data_augmentation
        self.num_workers = num_workers

        self.valid_stems = [
            f for f in os.listdir(data_dir) if f.endswith('.wav') and not analyze_audio(os.path.join(data_dir, f))[2]
        ]

        self.lock = Lock()
        global total_num_stems
        total_num_stems.value = len(self.valid_stems)
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.workers = self._start_workers()

    def _start_workers(self):
        workers = []
        stems_per_worker = len(self.valid_stems) // self.num_workers
        for i in range(self.num_workers):
            start_idx = i * stems_per_worker
            end_idx = start_idx + stems_per_worker if i < self.num_workers - 1 else len(self.valid_stems)
            worker_stems = self.valid_stems[start_idx:end_idx]
            p = Process(target=worker, args=(
                self.input_queue, self.output_queue, self.mel_spectrogram_params, self.data_dir, 
                self.target_length, self.apply_data_augmentation, worker_stems, self.lock, num_processed_stems
            ))
            p.start()
            workers.append(p)
        return workers

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, index):
        stem_name = self.valid_stems[index]
        self.input_queue.put((index, stem_name))

        while True:
            try:
                stem_name, data = self.output_queue.get(timeout=1)
                if data is not None:
                    return data
                logger.warning(f"Received None for stem_name {stem_name}. Retrying.")
                self.input_queue.put((index, stem_name))
            except queue.Empty:
                # Check if all stems have been processed
                with dataset_lock:
                    if num_processed_stems.value >= total_num_stems.value:
                        logger.info("All stems processed, ending iteration.")
                        raise StopIteration  # Signal the end of the dataset
                pass  # Continue waiting
            except (ConnectionResetError, EOFError) as e:
                logger.error(f"Connection error in __getitem__: {e}")
                self._restart_workers()
                self.input_queue.put((index, stem_name))

        return data

    def _restart_workers(self):
        logger.info("Restarting workers...")
        self.workers = self._start_workers()
        num_processed_stems.value = 0  # Reset the counter

if __name__ == "__main__":
    try:
        data_dir = "path/to/dataset"
        mel_spectrogram_params = {'sample_rate': 16000, 'n_mels': 64, 'n_fft': 1024}
        target_length = 256
        apply_data_augmentation = False

        dataset = StemSeparationDataset(data_dir, mel_spectrogram_params, target_length, apply_data_augmentation)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)  # num_workers=0 for compatibility with multiprocessing

        for data in dataloader:
            logger.info("Data loaded from DataLoader")

        logger.info("Processing complete.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
