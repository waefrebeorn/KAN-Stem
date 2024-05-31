from config import *
from train_utils import train_single_stem

def start_training():
    logger.info(f"Starting training with dataset at {data_dir}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    for stem in range(num_stems):
        train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_messages, suppress_reading_messages)

if __name__ == "__main__":
    start_training()
