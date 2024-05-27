import os
import logging
import warnings
from torch.utils.tensorboard import SummaryWriter
import gc

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

from model import KANWithDepthwiseConv, KANDiscriminator
from utils import detect_parameters
from log_setup import logger

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def read_audio(file_path):
    try:
        data, samplerate = sf.read(file_path)
        return torch.tensor(data).unsqueeze(0), samplerate
    except (FileNotFoundError, RuntimeError, sf.LibsndfileError) as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
        return None, None

def write_audio(file_path, data, samplerate):
    data = data.squeeze(0).cpu().numpy()
    sf.write(file_path, data, samplerate)

def check_device(model, *args):
    device = next(model.parameters()).device
    for idx, arg in enumerate(args):
        if arg.device != device:
            raise RuntimeError(
                f"Argument {idx} is on {arg.device}, but expected it to be on {device}"
            )

def custom_loss(output, target, loss_fn=nn.SmoothL1Loss()):
    if output.shape != target.shape:
        min_length = min(output.size(-1), target.size(-1))
        output = output[..., :min_length]
        target = target[..., :min_length]
    loss = loss_fn(output, target)
    return loss

def data_augmentation(inputs):
    # Define individual transforms
    pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2)
    freq_mask = T.FrequencyMasking(freq_mask_param=15)
    time_mask = T.TimeMasking(time_mask_param=35)
    
    # Apply transforms individually, creating new tensors each time
    augmented_inputs = pitch_shift(inputs.clone().detach())  # Create a copy of inputs to avoid in-place modification
    augmented_inputs = freq_mask(augmented_inputs.clone().detach())
    augmented_inputs = time_mask(augmented_inputs.clone().detach())
    
    return augmented_inputs

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels, n_fft=n_fft
        )
        self.valid_stems = [f for f in os.listdir(data_dir) if f.startswith("input") and f.endswith(".wav")]

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, index):
        stem_name = self.valid_stems[index]
        input_file = os.path.join(self.data_dir, stem_name)
        input_audio, _ = read_audio(input_file)
        if input_audio is None:
            return None

        input_audio = input_audio.float()
        if self.apply_data_augmentation:
            input_audio = data_augmentation(input_audio)
        input_mel = self.mel_spectrogram(input_audio).squeeze(0)[:, :self.target_length]

        parts = stem_name.split('_')
        if len(parts) < 2:
            logger.error(f"Invalid input file name format: {stem_name}")
            return None
        stem_id = parts[1].replace('.wav', '')
        
        target_stems = ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]
        target_mels = []
        for target_stem in target_stems:
            target_file = os.path.join(self.data_dir, f"target_{target_stem}_{stem_id}.wav")
            target_audio, _ = read_audio(target_file)
            if target_audio is None:
                return None
            target_audio = target_audio.float()
            target_mel = self.mel_spectrogram(target_audio).squeeze(0)[:, :self.target_length]
            target_mels.append(target_mel)

        return {'input': input_mel, 'target': torch.stack(target_mels)}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == "Momentum":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, lr=learning_rate)
    elif optimizer_name == "RMSProp":
        return optim.RMSprop(model_parameters, lr=learning_rate, alpha=0.99)
    elif optimizer_name == "Adadelta":
        return optim.Adadelta(model_parameters, lr=learning_rate)
    elif optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train_single_stem(stem, data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation=False):
    logger.info("Starting training for single stem: %s", stem)  # Log the specific stem being trained
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'))  # Separate log for each stem

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation)
    train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=use_cuda, num_workers=num_workers, collate_fn=collate_fn)

    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)  # num_stems=1 for single stem
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    criterion = loss_function
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=scheduler_step_size, gamma=scheduler_gamma)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, data in enumerate(train_loader):
            if data is None:
                continue

            inputs = data['input'].unsqueeze(1).to(device)
            targets = data['target'][:, stem].unsqueeze(1).to(device)  # Select the target stem

            batch_size = inputs.size(0)

            # Generator forward pass and backward pass
            outputs = model(inputs)  # Remove clone here, it's unnecessary as we clone in the line below.
            target_length = targets.size(-1)
            outputs = outputs[..., :target_length]  # Ensure matching length

            optimizer_g.zero_grad()
            loss_g = criterion(outputs, targets)
            if perceptual_loss_flag:
                # Add perceptual loss computation here if applicable
                pass
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer_g.step()
            running_loss_g += loss_g.item()

            # Discriminator forward pass and backward pass
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            real_out = discriminator(targets.clone().detach())
            fake_out = discriminator(outputs.clone().detach())

            loss_d_real = nn.BCEWithLogitsLoss()(real_out, real_labels)
            loss_d_fake = nn.BCEWithLogitsLoss()(fake_out, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
            optimizer_d.step()
            running_loss_d += loss_d.item()

            if i % accumulation_steps == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                writer.add_scalar('Loss/Generator', running_loss_g / accumulation_steps, epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Discriminator', running_loss_d / accumulation_steps, epoch * len(train_loader) + i)
                running_loss_g = 0.0
                running_loss_d = 0.0

            del outputs, targets
            gc.collect()
            torch.cuda.empty_cache()

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

    # Save final model for the stem
    final_model_path = f"{checkpoint_dir}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    writer.close()

def start_training(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation=False):
    logger.info(f"Starting training with dataset at {data_dir}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256  # Define your target length here

    for stem in range(num_stems):
        train_single_stem(stem, data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation)

if __name__ == "__main__":
    # Define training parameters
    data_dir = "path_to_data_dir"
    batch_size = 32
    num_epochs = 10
    learning_rate_g = 0.001
    learning_rate_d = 0.00005  # Reduced learning rate for the discriminator
    use_cuda = True
    checkpoint_dir = "path_to_checkpoint_dir"
    save_interval = 1
    accumulation_steps = 1
    num_stems = 7  # Adjusted to the number of stems in the dataset
    num_workers = 4
    cache_dir = "path_to_cache_dir"
    loss_function = nn.L1Loss()
    optimizer_name_g = "Adam"
    optimizer_name_d = "Adam"
    perceptual_loss_flag = True
    clip_value = 1.0
    scheduler_step_size = 5
    scheduler_gamma = 0.5

    # Start training
    start_training(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation=False)
