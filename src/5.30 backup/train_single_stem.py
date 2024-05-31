import gc
import os
import random
import warnings
import logging
import soundfile as sf
import queue  # Import the queue module for exception handling

import mir_eval
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from multiprocessing import Queue, Process
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchaudio
import torchaudio.transforms as T

from log_setup import logger
from model import KANWithDepthwiseConv, KANDiscriminator
from utils import detect_parameters, MultiScaleSpectralLoss, PerceptualLoss, compute_adversarial_loss

# Suppress specific warning
warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")

# Initialize hybrid cache with 32GB RAM limit
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
                return torch.load(file_path)
        return None

    def set(self, key, value):
        value_size = value.numel() * value.element_size()
        if self.current_ram_usage + value_size <= self.ram_limit:
            self.ram_cache[key] = value
            self.current_ram_usage += value_size
        else:
            file_path = os.path.join(self.ssd_cache_dir, key)
            torch.save(value, file_path)

cache = HybridCache(ram_limit=32 * 1024 ** 3, ssd_cache_dir='./ssd_cache')

def read_audio(file_path, device='cuda' if torch.cuda.is_available() else 'cpu', suppress_messages=False):
    try:
        if not suppress_messages:
            print(f"Attempting to read: {file_path}")
        data, samplerate = torchaudio.load(file_path)
        data = data.to(device)
        return data, samplerate
    except FileNotFoundError:
        logger.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
    return None, None

def write_audio(file_path, data, samplerate):
    data_cpu = data.squeeze(0).cpu().numpy()
    sf.write(file_path, data_cpu, samplerate)

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

def data_augmentation(inputs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2).to(device)
    freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
    time_mask = T.TimeMasking(time_mask_param=35).to(device)
    
    augmented_inputs = pitch_shift(inputs.clone().detach().to(device))
    augmented_inputs = freq_mask(augmented_inputs.clone().detach())
    augmented_inputs = time_mask(augmented_inputs.clone().detach())
    
    return augmented_inputs

def worker(input_queue, output_queue, mel_spectrogram_params_path, data_dir, target_length, device, apply_data_augmentation, suppress_messages):
    mel_spectrogram_params = torch.load(mel_spectrogram_params_path)
    mel_spectrogram = T.MelSpectrogram(sample_rate=16000, n_mels=mel_spectrogram_params['n_mels'], n_fft=mel_spectrogram_params['n_fft']).to(device)
    
    while True:
        try:
            index = input_queue.get(timeout=10)
            if index is None:
                break
            data = load_and_preprocess(index, mel_spectrogram, data_dir, target_length, device, apply_data_augmentation, suppress_messages)
            output_queue.put((index, data))
        except queue.Empty:
            logger.warning("Worker queue is empty, exiting worker.")
            break
        except Exception as e:
            logger.error(f"Worker encountered an error: {e}")
            output_queue.put((index, None))

def load_and_preprocess(index, mel_spectrogram, data_dir, target_length, device, apply_data_augmentation, suppress_messages):
    try:
        stem_name = os.listdir(data_dir)[index]
        input_file = os.path.join(data_dir, stem_name)
        input_audio, _ = read_audio(input_file, device='cpu', suppress_messages=suppress_messages)
        if input_audio is None:
            return None

        input_audio = input_audio.float()
        if apply_data_augmentation:
            input_audio = data_augmentation(input_audio, device=device)
        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]

        parts = stem_name.split('_')
        if len(parts) < 2:
            logger.error(f"Invalid input file name format: {stem_name}")
            return None
        stem_id = parts[1].replace('.wav', '')

        target_stems = ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]
        target_mels = []
        for target_stem in target_stems:
            target_audio = None
            target_file = os.path.join(data_dir, f"target_{target_stem}_{stem_id}.wav")

            # Keep searching for a valid substitute
            valid_target_files = [
                os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.startswith(f"target_{target_stem}_") and f.endswith(".wav") and f != target_file
            ]

            if not os.path.exists(target_file) or os.path.getsize(target_file) == 0:
                if valid_target_files:
                    valid_target_file = random.choice(valid_target_files)
                    logger.warning(f"Substituting {target_file} with {valid_target_file} for stem {target_stem}")
                    target_audio, _ = read_audio(valid_target_file, device='cpu', suppress_messages=suppress_messages)
                    if target_audio is not None:
                        wildcard_flag = torch.ones_like(target_audio) * -1  # Indicates this is a substitution
                        target_audio = torch.cat((target_audio, wildcard_flag), dim=-1)
                else:
                    logger.error(f"Error: No valid target found for stem '{target_stem}' and ID '{stem_id}'. Skipping this sample.")
                    return None
            else:
                target_audio, _ = read_audio(target_file, device='cpu', suppress_messages=suppress_messages)
                if target_audio is not None:
                    normal_flag = torch.zeros_like(target_audio)  # Indicates this is not a substitution
                    target_audio = torch.cat((target_audio, normal_flag), dim=-1)

            if target_audio is None:
                logger.error(f"Error: No valid target found for stem '{target_stem}' and ID '{stem_id}'. Skipping this sample.")
                return None

            # --- Check for silence ---
            if target_audio.numel() > 0 and torch.all(target_audio == 0):  # Check for all-zero tensors
                logger.warning(f"Skipping sample {stem_id} due to silent target stem: {target_stem}")
                wildcard_flag = torch.ones_like(target_audio) * -2  # Indicates silent substitution
                target_audio = torch.cat((target_audio, wildcard_flag), dim=-1)

            target_audio = target_audio.float()
            target_mel = mel_spectrogram(target_audio[..., :-1]).squeeze(0)[:, :target_length]  # Exclude the flag from mel spectrogram
            target_mels.append(target_mel)

        return {'input': input_mel, 'target': torch.stack(target_mels)}
    except Exception as e:
        logger.error(f"Error in load_and_preprocess: {e}")
        return None

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=False, num_workers=4):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_messages = suppress_messages
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_spectrogram_params = {'n_mels': n_mels, 'n_fft': n_fft}
        torch.save(self.mel_spectrogram_params, 'mel_spectrogram_params.pth')  # Save params to a file
        self.valid_stems = [f for f in os.listdir(data_dir) if f.startswith("input") and f.endswith(".wav")]

        if suppress_messages:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.INFO)

        # Multiprocessing
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        for _ in range(num_workers):
            p = Process(target=worker, args=(self.input_queue, self.output_queue, 'mel_spectrogram_params.pth', self.data_dir, self.target_length, self.device, self.apply_data_augmentation, self.suppress_messages))
            p.start()
            self.workers.append(p)

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, index):
        self.input_queue.put(index)
        while True:
            try:
                index, data = self.output_queue.get(timeout=10)  # Adjust timeout dynamically if needed
                if data is not None:
                    return data
            except queue.Empty:
                logger.warning("Queue is empty, waiting for data.")
                continue

    def __del__(self):
        for _ in range(len(self.workers)):
            self.input_queue.put(None)
        for p in self.workers:
            p.join()

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

def calculate_metrics(reference, estimation, sr=16000):
    reference_np = reference.squeeze().cpu().numpy()
    estimation_np = estimation.squeeze().cpu().numpy()

    # Check if reference is silent
    if np.all(reference_np == 0):
        logger.warning("Skipping silent reference audio in metrics calculation.")
        return np.nan, np.nan, np.nan  # Return NaN for silent references
    else:
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_np, estimation_np)
        return sdr, sir, sar

def train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_messages, suppress_reading_messages):
    logger.info("Starting training for single stem: %s", stem)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_messages, num_workers)
    train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataset = StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=suppress_messages, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d, weight_decay)

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=num_epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=num_epochs)

    scaler = GradScaler()
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for i, data in enumerate(train_loader):
            if data is None:
                continue

            # --- GPU Memory Management ---
            _, available_memory = torch.cuda.mem_get_info()
            input_size = data['input'].numel() * data['input'].element_size()
            target_size = data['target'][:, stem].numel() * data['target'][:, stem].element_size()
            total_size = input_size + target_size

            # --- Load to GPU/CPU based on priority ---
            inputs = data['input'].unsqueeze(1)
            targets = data['target'][:, stem].unsqueeze(1)

            # Prioritize inputs, then targets
            if available_memory > total_size:
                inputs = inputs.to(device)
                targets = targets.to(device)
            elif available_memory > input_size:
                inputs = inputs.to(device)
                targets = targets.to('cpu')
            else:
                inputs = inputs.to('cpu')
                targets = targets.to('cpu')

            # --- Offloading & Checkpointing ---
            if available_memory < (input_size + target_size):
                # Offload less critical tensors to CPU RAM
                model.to('cpu')
                # ... (Perform operations on CPU)
                model.to(device)

            # Use torch.utils.checkpoint for memory-intensive parts of your model
            with autocast():
                outputs = model(inputs.to(device))
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                loss_g = loss_function_g(outputs, targets.to(device))
                scaler.scale(loss_g).backward()

                real_labels = torch.ones(inputs.size(0), 1, device=device)
                fake_labels = torch.zeros(inputs.size(0), 1, device=device)
                real_out = discriminator(targets.to(device).clone().detach())
                fake_out = discriminator(outputs.clone().detach())

                loss_d_real = loss_function_d(real_out, real_labels)
                loss_d_fake = loss_function_d(fake_out, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                scaler.scale(loss_d).backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                    scaler.step(optimizer_g)
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()

                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                    if tensorboard_flag:
                        writer.add_scalar('Loss/Generator', running_loss_g / accumulation_steps, epoch * len(train_loader) + i)
                        writer.add_scalar('Loss/Discriminator', running_loss_d / accumulation_steps, epoch * len(train_loader) + i)
                    running_loss_g = 0.0
                    running_loss_d = 0.0

            del inputs, targets, outputs
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
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
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

                for j in range(outputs.size(0)):
                    sdr, sir, sar = calculate_metrics(targets[j], outputs[j])
                    if not np.isnan(sdr):
                        sdr_total += sdr.mean()
                        sir_total += sir.mean()
                        sar_total += sar.mean()

        # Calculate averages, excluding NaN values
        num_valid_samples = len(val_loader.dataset) - np.isnan(sdr_total).sum()  # Exclude silent references
        val_loss /= len(val_loader)
        sdr_avg = sdr_total / num_valid_samples
        sir_avg = sir_total / num_valid_samples
        sar_avg = sar_total / num_valid_samples

        logger.info(f'Validation Loss: {val_loss:.4f}, SDR: {sdr_avg:.4f}, SIR: {sir_avg:.4f}, SAR: {sar_avg:.4f}')
        if tensorboard_flag:
            writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            writer.add_scalar('Metrics/SDR', sdr_avg, epoch + 1)
            writer.add_scalar('Metrics/SIR', sir_avg, epoch + 1)
            writer.add_scalar('Metrics/SAR', sar_avg, epoch + 1)

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

def start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_messages, suppress_reading_messages):
    logger.info(f"Starting training with dataset at {data_dir}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    for stem in range(num_stems):
        train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_messages, suppress_reading_messages)

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
    suppress_messages = False
    suppress_reading_messages = False

    start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_messages, suppress_reading_messages)
