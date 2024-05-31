import os
import torch
from torch.utils.data import DataLoader
from train import start_training_wrapper
from dataset import StemSeparationDataset, collate_fn

if __name__ == "__main__":
    data_dir = r"K:\KAN-Stem DataSet\ProcessedDataset"
    val_dir = r"K:\KAN-Stem DataSet\ValidationDataset"
    n_mels = 64
    target_length = 128
    n_fft = 1024
    cache_dir = r"K:\KAN-Stem DataSet\Cache"
    apply_data_augmentation = True
    suppress_warnings = False
    num_workers = 4
    batch_size = 8
    num_epochs = 10
    learning_rate_g = 0.001
    learning_rate_d = 0.001
    use_cuda = True
    checkpoint_dir = r"K:\KAN-Stem DataSet\Checkpoints"
    save_interval = 5
    accumulation_steps = 4
    num_stems = 1
    loss_function_str_g = "MSELoss"
    loss_function_str_d = "BCEWithLogitsLoss"
    optimizer_name_g = "Adam"
    optimizer_name_d = "Adam"
    perceptual_loss_flag = False
    clip_value = 1.0
    scheduler_step_size = 5
    scheduler_gamma = 0.5
    tensorboard_flag = True
    add_noise = False
    noise_amount = 0.01
    early_stopping_patience = 3
    weight_decay = 0.0001
    suppress_reading_messages = True
    use_cpu_for_prep = True

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    # Create the dataset objects
    train_dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device)
    val_dataset = StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, False, suppress_warnings, num_workers, device)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep)
