import os
import logging
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import gc
import random

from model import KANWithDepthwiseConv, KANDiscriminator
from utils import detect_parameters
from log_setup import logger

class HybridCache:
    def __init__(self, ram_limit=32 * 1024 ** 3, ssd_cache_dir='./ssd_cache'):
        self.ram_limit = ram_limit
        self.ssd_cache_dir = ssd_cache_dir
        self.current_ram_usage = 0
        self.ram_cache = {}
        if not os.path.exists(ssd_cache_dir):
            os.makedirs(ssd_cache_dir)

    def get(self, key):
        if key in self.ram_cache:
            return self.ram_cache[key]
        else:
            file_path = os.path.join(self.ssd_cache_dir, key)
            if os.path.exists(file_path):
                return np.load(file_path)
        return None

    def set(self, key, value):
        value_size = value.nbytes
        if self.current_ram_usage + value_size <= self.ram_limit:
            self.ram_cache[key] = value
            self.current_ram_usage += value_size
        else:
            file_path = os.path.join(self.ssd_cache_dir, key)
            np.save(file_path, value)

# Initialize hybrid cache with 32GB RAM limit
cache = HybridCache(ram_limit=32 * 1024 ** 3, ssd_cache_dir='./ssd_cache')

def read_audio(file_path):
    try:
        print(f"Attempting to read: {file_path}")
        data, samplerate = sf.read(file_path)
        return torch.tensor(data).unsqueeze(0), samplerate
    except FileNotFoundError:
        logger.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
    except sf.LibsndfileError as e:
        logger.error(f"Error decoding audio file {file_path}: {e}")
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
    pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2)
    freq_mask = T.FrequencyMasking(freq_mask_param=15)
    time_mask = T.TimeMasking(time_mask_param=35)
    
    augmented_inputs = pitch_shift(inputs.clone().detach())
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
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=16000, n_mels=n_mels, n_fft=n_fft)
        self.valid_stems = [f for f in os.listdir(data_dir) if f.startswith("input") and f.endswith(".wav")]
        self.available_targets = self._get_available_targets()

    def __len__(self):
        return len(self.valid_stems)

    def _get_available_targets(self):
        targets = {}
        for target_stem in ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]:
            targets[target_stem] = []
            for stem_file in self.valid_stems:
                stem_id = stem_file.split('_')[1].replace('.wav', '')
                target_file = os.path.join(self.data_dir, f"target_{target_stem}_{stem_id}.wav")
                if os.path.exists(target_file):
                    targets[target_stem].append(target_file)
        return targets

    def _get_random_target(self, target_stem):
        return random.choice(self.available_targets[target_stem])

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
                # Substitute with a random existing target if the file is missing
                target_file = self._get_random_target(target_stem)
                target_audio, _ = read_audio(target_file)
                if self.apply_data_augmentation:
                    target_audio = data_augmentation(target_audio)

            target_audio = target_audio.float()
            target_mel = self.mel_spectrogram(target_audio).squeeze(0)[:, :self.target_length]
            target_mels.append(target_mel)

        return {'input': input_mel, 'target': torch.stack(target_mels)}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

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

def train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay):
    logger.info("Starting training for single stem: %s", stem)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation)
    train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=use_cuda, num_workers=num_workers, collate_fn=collate_fn)

    val_dataset = StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=use_cuda, num_workers=num_workers, collate_fn=collate_fn)

    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d, weight_decay)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=scheduler_step_size, gamma=scheduler_gamma)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=scheduler_step_size, gamma=scheduler_gamma)

    scaler = GradScaler()
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, data in enumerate(train_loader):
            if data is None:
                continue

            inputs = data['input'].unsqueeze(1).to(device)
            targets = data['target'][:, stem].unsqueeze(1).to(device)
            batch_size = inputs.size(0)

            if add_noise:
                noise = torch.randn_like(inputs) * noise_amount
                inputs = inputs + noise

            with autocast():
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                optimizer_g.zero_grad()
                loss_g = loss_function_g(outputs, targets)
                if perceptual_loss_flag:
                    pass
                scaler.scale(loss_g).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer_g)
                scaler.update()
                running_loss_g += loss_g.item()

                optimizer_d.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                real_out = discriminator(targets.clone().detach())
                fake_out = discriminator(outputs.clone().detach())

                loss_d_real = loss_function_d(real_out, real_labels)
                loss_d_fake = loss_function_d(fake_out, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                scaler.scale(loss_d).backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                scaler.step(optimizer_d)
                scaler.update()
                running_loss_d += loss_d.item()

            if i % accumulation_steps == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                if tensorboard_flag:
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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue
                inputs = data['input'].unsqueeze(1).to(device)
                targets = data['target'][:, stem].unsqueeze(1).to(device)
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]
                loss = loss_function_g(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        if tensorboard_flag:
            writer.add_scalar('Loss/Validation', val_loss, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            logger.info('Early stopping triggered.')
            break

    final_model_path = f"{checkpoint_dir}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if tensorboard_flag:
        writer.close()

def start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay):
    logger.info(f"Starting training with dataset at {data_dir}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    for stem in range(num_stems):
        train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay)

if __name__ == "__main__":
    data_dir = "path_to_data_dir"
    val_dir = "path_to_validation_dir"
    batch_size = 32
    num_epochs = 10
    learning_rate_g = 0.001
    learning_rate_d = 0.00005
    use_cuda = True
    checkpoint_dir = "path_to_checkpoint_dir"
    save_interval = 1
    accumulation_steps = 1
    num_stems = 7
    num_workers = 4
    cache_dir = "path_to_cache_dir"
    loss_function_g = nn.L1Loss()
    loss_function_d = nn.BCEWithLogitsLoss()
    optimizer_name_g = "Adam"
    optimizer_name_d = "Adam"
    perceptual_loss_flag = True
    clip_value = 1.0
    scheduler_step_size = 5
    scheduler_gamma = 0.5
    tensorboard_flag = True
    apply_data_augmentation = False
    add_noise = False
    noise_amount = 0.1
    early_stopping_patience = 3
    weight_decay = 1e-4

    start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay)
