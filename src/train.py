import torch
import torch.multiprocessing as mp
import torch.nn as nn
import logging
from data_preprocessing import preprocess_and_cache_dataset
from training_loop import train_single_stem
from loss_functions import wasserstein_loss
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import librosa
from ray.air import session

logger = logging.getLogger(__name__)

mp.set_start_method('spawn', force=True)

def detect_parameters(data_dir):
    sample_rates = []
    n_mels = 128  # Default value or derived from specific needs
    n_fft = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)
                sample_rates.append(sr)
                n_fft.append(len(y) // 2)

    if not sample_rates or not n_fft:
        raise ValueError(f"No audio files found in the specified data directory: {data_dir}")

    sample_rate = max(set(sample_rates), key=sample_rates.count)
    n_fft = max(set(n_fft), key=n_fft.count)
    
    return sample_rate, n_mels, n_fft

def log_training_parameters(params):
    for key, value in params.items():
        logger.info(f"{key}: {value}")

def start_training(data_dir, val_dir, batch_size, num_epochs, initial_lr_g, initial_lr_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs):
    global device

    logger.info(f"Starting training with dataset at {data_dir}")
    device_str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    device_prep = 'cpu' if use_cpu_for_prep else device_str
    logger.info(f"Using device: {device_str} for training and {device_prep} for preprocessing")

    try:
        sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    except ValueError as e:
        logger.error(f"Error in detecting parameters: {e}")
        return

    target_length = 256

    preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep)
    
    params = {
        "Data Directory": data_dir,
        "Validation Directory": val_dir,
        "Batch Size": batch_size,
        "Number of Epochs": num_epochs,
        "Generator Learning Rate": initial_lr_g,
        "Discriminator Learning Rate": initial_lr_d,
        "Use CUDA": use_cuda,
        "Checkpoint Directory": checkpoint_dir,
        "Save Interval": save_interval,
        "Accumulation Steps": accumulation_steps,
        "Number of Stems": num_stems,
        "Number of Workers": num_workers,
        "Cache Directory": cache_dir,
        "Generator Loss Function": str(loss_function_g),
        "Discriminator Loss Function": str(loss_function_d),
        "Generator Optimizer": optimizer_name_g,
        "Discriminator Optimizer": optimizer_name_d,
        "Use Perceptual Loss": perceptual_loss_flag,
        "Perceptual Loss Weight": perceptual_loss_weight,
        "Gradient Clipping Value": clip_value,
        "Scheduler Step Size": scheduler_step_size,
        "Scheduler Gamma": scheduler_gamma,
        "Enable TensorBoard Logging": tensorboard_flag,
        "Apply Data Augmentation": apply_data_augmentation,
        "Add Noise": add_noise,
        "Noise Amount": noise_amount,
        "Early Stopping Patience": early_stopping_patience,
        "Disable Early Stopping": disable_early_stopping,
        "Weight Decay": weight_decay,
        "Suppress Warnings": suppress_warnings,
        "Suppress Reading Messages": suppress_reading_messages,
        "Use CPU for Preparation": use_cpu_for_prep,
        "Discriminator Update Interval": discriminator_update_interval,
        "Label Smoothing Real": label_smoothing_real,
        "Label Smoothing Fake": label_smoothing_fake,
        "Suppress Detailed Logs": suppress_detailed_logs
    }
    log_training_parameters(params)

    if scheduler_gamma >= 1.0:
        scheduler_gamma = 0.9
        logger.warning("scheduler_gamma should be < 1.0. Setting scheduler_gamma to 0.9")

    for stem in range(num_stems):
        try:
            train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, device_str, checkpoint_dir, save_interval, accumulation_steps, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, loss_function_g, loss_function_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, num_workers, early_stopping_patience, disable_early_stopping, weight_decay, use_cpu_for_prep, suppress_warnings, suppress_reading_messages, cache_dir, device_prep, add_noise, noise_amount, discriminator_update_interval, label_smoothing_real, label_smoothing_fake)
        except Exception as e:
            logger.error(f"Error in train_single_stem: {e}", exc_info=True)
            raise

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs):
    global training_process
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "WassersteinLoss": wasserstein_loss
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]
    training_process = mp.Process(target=start_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"
    
def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

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

def objective_optuna(trial):
    batch_size = trial.suggest_int('batch_size', 16, 64)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    learning_rate_g = trial.suggest_float('learning_rate_g', 1e-5, 1e-1, log=True)
    learning_rate_d = trial.suggest_float('learning_rate_d', 1e-5, 1e-1, log=True)
    perceptual_loss_weight = trial.suggest_float('perceptual_loss_weight', 0.0, 1.0)
    clip_value = trial.suggest_float('clip_value', 0.5, 1.5)

    start_training(data_dir="K:/KAN-Stem DataSet/ProcessedDataset", val_dir="K:/KAN-Stem DataSet/Chunk_0_Sample", batch_size=batch_size, num_epochs=num_epochs,
                   initial_lr_g=learning_rate_g, initial_lr_d=learning_rate_d, use_cuda=True,
                   checkpoint_dir="./checkpoints", save_interval=50, accumulation_steps=4, num_stems=7,
                   num_workers=4, cache_dir="./cache", loss_function_g=nn.L1Loss(), loss_function_d=wasserstein_loss,
                   optimizer_name_g="Adam", optimizer_name_d="Adam", perceptual_loss_flag=True,
                   perceptual_loss_weight=perceptual_loss_weight, clip_value=clip_value, scheduler_step_size=10,
                   scheduler_gamma=0.9, tensorboard_flag=False, apply_data_augmentation=True, add_noise=True,
                   noise_amount=0.1, early_stopping_patience=3, disable_early_stopping=False, weight_decay=1e-4,
                   suppress_warnings=True, suppress_reading_messages=True, use_cpu_for_prep=False,
                   discriminator_update_interval=5, label_smoothing_real=0.8, label_smoothing_fake=0.2,
                   suppress_detailed_logs=True)

    return 0.0

def train_ray_tune(config):
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate_g = config['learning_rate_g']
    learning_rate_d = config['learning_rate_d']
    perceptual_loss_weight = config['perceptual_loss_weight']
    clip_value = config['clip_value']

    start_training(data_dir="K:/KAN-Stem DataSet/ProcessedDataset", val_dir="K:/KAN-Stem DataSet/Chunk_0_Sample", batch_size=batch_size, num_epochs=num_epochs,
                   initial_lr_g=learning_rate_g, initial_lr_d=learning_rate_d, use_cuda=True,
                   checkpoint_dir="./checkpoints", save_interval=50, accumulation_steps=4, num_stems=7,
                   num_workers=4, cache_dir="./cache", loss_function_g=nn.L1Loss(), loss_function_d=wasserstein_loss,
                   optimizer_name_g="Adam", optimizer_name_d="Adam", perceptual_loss_flag=True,
                   perceptual_loss_weight=perceptual_loss_weight, clip_value=clip_value, scheduler_step_size=10,
                   scheduler_gamma=0.9, tensorboard_flag=False, apply_data_augmentation=True, add_noise=True,
                   noise_amount=0.1, early_stopping_patience=3, disable_early_stopping=False, weight_decay=1e-4,
                   suppress_warnings=True, suppress_reading_messages=True, use_cpu_for_prep=False,
                   discriminator_update_interval=5, label_smoothing_real=0.8, label_smoothing_fake=0.2,
                   suppress_detailed_logs=True)

    session.report({"metric": 0.0})

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_optuna, n_trials=10)

    ray.init()
    config = {
        'batch_size': tune.choice([16, 32, 64]),
        'num_epochs': tune.choice([10, 20, 50, 100]),
        'learning_rate_g': tune.loguniform(1e-5, 1e-1),
        'learning_rate_d': tune.loguniform(1e-5, 1e-1),
        'perceptual_loss_weight': tune.uniform(0.0, 1.0),
        'clip_value': tune.uniform(0.5, 1.5)
    }
    scheduler = ASHAScheduler(
        metric='metric',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    tune.run(train_ray_tune, config=config, scheduler=scheduler, num_samples=10)
