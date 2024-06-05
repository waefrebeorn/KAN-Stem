import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import logging
from data_preprocessing import preprocess_and_cache_dataset
from training_loop import train_single_stem
from loss_functions import wasserstein_loss
from dataset import StemSeparationDataset
from utils import log_training_parameters, detect_parameters
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

logger = logging.getLogger(__name__)

mp.set_start_method('spawn', force=True)

# Define the global variable `training_process` and `stop_flag`
training_process = None
stop_flag = mp.Value('i', 0)  # Shared flag for stopping processes

def start_training(data_dir, val_dir, training_params, model_params, stop_flag):
    if training_params['suppress_warnings']:
        logger.setLevel(logging.ERROR)
    elif training_params['suppress_reading_messages']:
        logger.setLevel(logging.WARNING)

    logger.info(f"Starting training with dataset at {data_dir}")
    device_str = 'cuda' if training_params['use_cuda'] and torch.cuda.is_available() else 'cpu'
    training_params['device_str'] = device_str
    device_prep = 'cpu' if training_params['use_cpu_for_prep'] else device_str
    logger.info(f"Using device: {device_str} for training and {device_prep} for preprocessing")

    try:
        sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    except ValueError as e:
        logger.error(f"Error in detecting parameters: {e}")
        return

    target_length = 256

    preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], training_params['apply_data_augmentation'], training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device_prep, stop_flag)
    
    if stop_flag.value == 1:
        logger.info("Training stopped during data preprocessing.")
        return

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], training_params['apply_data_augmentation'], training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device_prep, stop_flag)
    
    params = {**training_params, **model_params}
    log_training_parameters(params)

    if model_params['scheduler_gamma'] >= 1.0:
        model_params['scheduler_gamma'] = 0.9
        logger.warning("scheduler_gamma should be < 1.0. Setting scheduler_gamma to 0.9")

    for stem in range(training_params['num_stems']):
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return
        try:
            train_single_stem(stem, dataset, val_dir, training_params, model_params, sample_rate, n_mels, n_fft, target_length, stop_flag)
        except Exception as e:
            logger.error(f"Error in train_single_stem: {e}", exc_info=True)
            raise

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake):
    global training_process, stop_flag
    stop_flag.value = 0  # Reset stop flag
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "WassersteinLoss": wasserstein_loss
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]

    training_params = {
        'data_dir': data_dir,
        'val_dir': val_dir,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'use_cuda': use_cuda,
        'checkpoint_dir': checkpoint_dir,
        'save_interval': save_interval,
        'accumulation_steps': accumulation_steps,
        'num_stems': num_stems,
        'num_workers': num_workers,
        'cache_dir': cache_dir,
        'apply_data_augmentation': apply_data_augmentation,
        'add_noise': add_noise,
        'noise_amount': noise_amount,
        'early_stopping_patience': early_stopping_patience,
        'disable_early_stopping': disable_early_stopping,
        'suppress_warnings': suppress_warnings,
        'suppress_reading_messages': suppress_reading_messages,
        'use_cpu_for_prep': use_cpu_for_prep,
        'discriminator_update_interval': discriminator_update_interval,
        'label_smoothing_real': label_smoothing_real,
        'label_smoothing_fake': label_smoothing_fake
    }

    model_params = {
        'initial_lr_g': learning_rate_g,
        'initial_lr_d': learning_rate_d,
        'loss_function_g': loss_function_g,
        'loss_function_d': loss_function_d,
        'optimizer_name_g': optimizer_name_g,
        'optimizer_name_d': optimizer_name_d,
        'perceptual_loss_flag': perceptual_loss_flag,
        'perceptual_loss_weight': perceptual_loss_weight,
        'clip_value': clip_value,
        'scheduler_step_size': scheduler_step_size,
        'scheduler_gamma': scheduler_gamma,
        'tensorboard_flag': tensorboard_flag,
        'weight_decay': weight_decay
    }

    training_process = mp.Process(target=start_training, args=(data_dir, val_dir, training_params, model_params, stop_flag))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"

def stop_training_wrapper():
    global training_process, stop_flag
    if training_process is not None:
        stop_flag.value = 1  # Set stop flag to request stopping
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def resume_training(checkpoint_dir, device_str):
    logger.info(f"Resuming training from checkpoint at {checkpoint_dir}")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        return "No checkpoints found in the specified directory."

    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device_str)
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
        return f"Error loading checkpoint: {e}"

    # Assuming the checkpoint contains model and optimizer state
    model = initialize_model()
    optimizer = initialize_optimizer()

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    start_training(
        data_dir=checkpoint['data_dir'],
        val_dir=checkpoint['val_dir'],
        batch_size=checkpoint['batch_size'],
        num_epochs=checkpoint['num_epochs'] - epoch,
        initial_lr_g=checkpoint['initial_lr_g'],
        initial_lr_d=checkpoint['initial_lr_d'],
        use_cuda='cuda' in device_str,
        checkpoint_dir=checkpoint['checkpoint_dir'],
        save_interval=checkpoint['save_interval'],
        accumulation_steps=checkpoint['accumulation_steps'],
        num_stems=checkpoint['num_stems'],
        num_workers=checkpoint['num_workers'],
        cache_dir=checkpoint['cache_dir'],
        loss_function_g=checkpoint['loss_function_g'],
        loss_function_d=checkpoint['loss_function_d'],
        optimizer_name_g=checkpoint['optimizer_name_g'],
        optimizer_name_d=checkpoint['optimizer_name_d'],
        perceptual_loss_flag=checkpoint['perceptual_loss_flag'],
        perceptual_loss_weight=checkpoint['perceptual_loss_weight'],
        clip_value=checkpoint['clip_value'],
        scheduler_step_size=checkpoint['scheduler_step_size'],
        scheduler_gamma=checkpoint['scheduler_gamma'],
        tensorboard_flag=checkpoint['tensorboard_flag'],
        apply_data_augmentation=checkpoint['apply_data_augmentation'],
        add_noise=checkpoint['add_noise'],
        noise_amount=checkpoint['noise_amount'],
        early_stopping_patience=checkpoint['early_stopping_patience'],
        disable_early_stopping=checkpoint['disable_early_stopping'],
        weight_decay=checkpoint['weight_decay'],
        suppress_warnings=checkpoint['suppress_warnings'],
        suppress_reading_messages=checkpoint['suppress_reading_messages'],
        use_cpu_for_prep=checkpoint['use_cpu_for_prep'],
        discriminator_update_interval=checkpoint['discriminator_update_interval'],
        label_smoothing_real=checkpoint['label_smoothing_real'],
        label_smoothing_fake=checkpoint['label_smoothing_fake']
    )

    return f"Resumed training from checkpoint: {latest_checkpoint}"

def resume_training_wrapper(checkpoint_dir):
    global training_process
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_process = mp.Process(target=resume_training, args=(checkpoint_dir, device_str))
    training_process.start()
    return f"Resuming training from checkpoint in {checkpoint_dir}"
