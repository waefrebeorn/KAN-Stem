import os
import torch
import torch.optim as optim
from multiprocessing import Value, Process
from model_setup import create_model_and_optimizer, initialize_model
from training_loop import start_training
import logging
from utils import log_training_parameters, detect_parameters, get_optimizer
from loss_functions import wasserstein_loss
import torch.nn as nn

logger = logging.getLogger(__name__)

# Define global variables
stop_flag = Value('i', 0)
training_process = None

def start_training_wrapper(
    data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
    accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g,
    optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma,
    tensorboard_flag, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay,
    suppress_warnings, suppress_reading_messages, discriminator_update_interval, label_smoothing_real, label_smoothing_fake,
    suppress_detailed_logs, use_cache, channel_multiplier, segments_per_track=10, update_cache=True
):
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

    training_process = Process(target=start_training, args=(
        data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
        accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
        perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise,
        noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
        discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, stop_flag, use_cache, channel_multiplier, segments_per_track, update_cache))
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
    model = initialize_model(device_str, checkpoint['n_mels'], checkpoint['target_length'])
    optimizer = get_optimizer(checkpoint['optimizer_name_g'], model.parameters(), checkpoint['initial_lr_g'], checkpoint['weight_decay'])

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    start_training(
        data_dir=checkpoint['data_dir'],
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
        add_noise=checkpoint['add_noise'],
        noise_amount=checkpoint['noise_amount'],
        early_stopping_patience=checkpoint['early_stopping_patience'],
        disable_early_stopping=checkpoint['disable_early_stopping'],
        weight_decay=checkpoint['weight_decay'],
        suppress_warnings=checkpoint['suppress_warnings'],
        suppress_reading_messages=checkpoint['suppress_reading_messages'],
        channel_multiplier=checkpoint['channel_multiplier'],
        segments_per_track=checkpoint.get('segments_per_track', 10),
        stop_flag=stop_flag,
        update_cache=checkpoint.get('update_cache', True)
    )

    return f"Resumed training from checkpoint: {latest_checkpoint}"

def resume_training_wrapper(checkpoint_dir):
    global training_process
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_process = Process(target=resume_training, args=(checkpoint_dir, device_str))
    training_process.start()
    return f"Resuming training from checkpoint in {checkpoint_dir}"

if __name__ == "__main__":
    # Example usage (optional, for testing)
    data_dir = "path/to/data"
    batch_size = 16
    num_epochs = 100
    learning_rate_g = 1e-4
    learning_rate_d = 1e-4
    use_cuda = torch.cuda.is_available()
    checkpoint_dir = "path/to/checkpoints"
    save_interval = 10
    accumulation_steps = 1
    num_stems = 4
    num_workers = 4
    cache_dir = "path/to/cache"
    loss_function_str_g = "MSELoss"
    loss_function_str_d = "BCEWithLogitsLoss"
    optimizer_name_g = "Adam"
    optimizer_name_d = "RMSprop"
    perceptual_loss_flag = True
    perceptual_loss_weight = 0.1
    clip_value = 0.01
    scheduler_step_size = 10
    scheduler_gamma = 0.1
    tensorboard_flag = True
    add_noise = True
    noise_amount = 0.1
    early_stopping_patience = 10
    disable_early_stopping = False
    weight_decay = 1e-5
    suppress_warnings = True
    suppress_reading_messages = False
    discriminator_update_interval = 5
    label_smoothing_real = 0.9
    label_smoothing_fake = 0.1
    suppress_detailed_logs = False
    use_cache = True
    channel_multiplier = 1.0
    segments_per_track = 10
    update_cache = True

    start_training_wrapper(
        data_dir,  batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, 
        accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d,
        perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag,
        add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
        discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, use_cache, channel_multiplier, segments_per_track, update_cache
    )
