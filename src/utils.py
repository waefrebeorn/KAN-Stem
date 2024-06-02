import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F  # Add this import
import logging
import soundfile as sf
import psutil
import GPUtil
import time
import torch.nn as nn
from multiprocessing import Value, Lock

logger = logging.getLogger(__name__)

num_processed_stems = Value('i', 0)
total_num_stems = Value('i', 0)
dataset_lock = Lock()

def analyze_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / sample_rate
        is_silent = waveform.abs().max() == 0
        return sample_rate, duration, is_silent
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        return None, None, True

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
    real_labels = torch.ones(y_true.size(0), 1, device=device)
    fake_labels = torch.zeros(y_pred.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy_with_logits(discriminator(y_true), real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(y_pred), fake_labels)
    return (real_loss + fake_loss) / 2

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

def write_audio(file_path, data, samplerate):
    try:
        data_cpu = data.squeeze(0).cpu().numpy()
        sf.write(file_path, data_cpu, samplerate)
    except Exception as e:
        logger.error(f"Error writing audio file {file_path}: {e}")

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

def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.debug("Starting load and preprocess")
        input_audio, _ = read_audio(file_path, device='cpu')
        if input_audio is None:
            return None

        logger.debug(f"Device for processing: {device}")

        if apply_data_augmentation:
            input_audio = input_audio.float().to(device)
            logger.debug(f"input_audio device: {input_audio.device}")
            input_audio = data_augmentation(input_audio, device=device)
            logger.debug(f"After data augmentation, input_audio device: {input_audio.device}")
        else:
            input_audio = input_audio.float().to(device)

        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]

        # Move input_mel back to CPU after processing on GPU
        input_mel = input_mel.to('cpu')
        logger.debug(f"input_mel device: {input_mel.device}")

        logger.debug("Completed load and preprocess")
        return input_mel

    except Exception as e:
        logger.error(f"Error in load and preprocess: {e}")
        return None

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
        logger.error(f"Error in calculating metrics: {e}")
        return {'mse': None, 'snr': None}

def get_checkpoints(checkpoints_dir):
    """Retrieve all checkpoint files in the specified directory."""
    try:
        checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints
    except Exception as e:
        logger.error(f"Error retrieving checkpoints: {e}")
        return []
