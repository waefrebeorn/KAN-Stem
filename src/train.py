import os
import torch
import torch.optim as optim
from multiprocessing import Value
from model_setup import create_model_and_optimizer
from training_loop import train_single_stem, start_training
import logging
from data_preprocessing import preprocess_and_cache_dataset
from dataset import StemSeparationDataset
from utils import log_training_parameters, detect_parameters

logger = logging.getLogger(__name__)

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs):
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
        'label_smoothing_fake': label_smoothing_fake,
        'suppress_detailed_logs': suppress_detailed_logs
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

    training_process = mp.Process(target=start_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs))
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
