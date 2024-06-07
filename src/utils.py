import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import soundfile as sf
import psutil
import GPUtil
import time
import torch.nn as nn
from torch.autograd import grad
from multiprocessing import Value, Lock
import torchaudio.transforms as T
from preprocessing_utils import load_and_preprocess

import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

logger = logging.getLogger(__name__)

num_processed_stems = Value('i', 0)
total_num_stems = Value('i', 0)
dataset_lock = Lock()

def setup_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def detect_parameters(data_dir, default_n_mels=64, default_n_fft=1024):
    sample_rates = []
    durations = []

    # Commented out to reduce console output
    # logger.info(f"Contents of the data directory ({data_dir}): {os.listdir(data_dir)}")

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

def analyze_audio(file_path):
    try:
        data, sample_rate = sf.read(file_path)
        duration = len(data) / sample_rate
        is_silent = (abs(data).max() < 1e-5)
        return sample_rate, duration, is_silent
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        return None, None, True

def convert_to_3_channels(tensor):
    """Convert a single-channel tensor to a 3-channel tensor by repeating the single channel."""
    return tensor.repeat(1, 3, 1, 1)

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

def compute_adversarial_loss(discriminator, y_true, y_pred, device):
    y_true = y_true.unsqueeze(1) if y_true.dim() == 3 else y_true
    y_pred = y_pred.unsqueeze(1) if y_pred.dim() == 3 else y_pred
    real_labels = torch.ones(y_true.size(0), 1, device=device)
    fake_labels = torch.zeros(y_pred.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy_with_logits(discriminator(y_true), real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(y_pred), fake_labels)
    return real_loss, fake_loss

def gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.to(device)
    
    interpolated.requires_grad_(True)
    
    prob_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

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

def process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.info(f"Processing file: {file_path}")
        log_system_resources()
        start_time = time.time()

        result = load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device)
        if result is not None:
            pass

        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds")

        logger.debug(f"Post-processing check for file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")

def worker_loop(file_paths, mel_spectrogram_params, target_length, apply_data_augmentation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params).to(device)

    try:
        for file_path in file_paths:
            logger.debug(f"Starting processing for file: {file_path}")
            process_file(file_path, mel_spectrogram, target_length, apply_data_augmentation, device)
            logger.debug(f"Finished processing for file: {file_path}")
        logger.debug("All files processed in worker loop")

        logger.debug("Entering post-processing stage after worker loop")
        logger.debug("Completed post-processing stage")

    except Exception as e:
        logger.error(f"Error in worker loop: {e}")

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for audio prediction."""
    try:
        mse = F.mse_loss(y_pred, y_true).item()
        snr = 10 * torch.log10(torch.mean(y_true ** 2) / torch.mean((y_true - y_pred) ** 2)).item()
        return {'mse': mse, 'snr': snr}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        raise e

def get_checkpoints(checkpoints_dir):
    """Retrieve all checkpoint files in the specified directory."""
    try:
        checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints
    except Exception as e:
        logger.error(f"Error retrieving checkpoints: {e}")
        return []

def compute_sdr(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    sdr = 10 * torch.log10(s_true / (s_noise + 1e-8))
    return sdr

def compute_sir(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_interf = torch.sum((true - noise) ** 2, dim=[1, 2, 3])
    sir = 10 * torch.log10(s_true / (s_interf + 1e-8))
    return sir

def compute_sar(true, pred):
    noise = true - pred
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    s_artif = torch.sum((pred - noise) ** 2, dim=[1, 2, 3])
    sar = 10 * torch.log10(s_noise / (s_artif + 1e-8))
    return sar

def log_training_parameters(params):
    """Logs the training parameters."""
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
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

def wasserstein_loss(real_output, fake_output):
    return torch.mean(fake_output) - torch.mean(real_output)
