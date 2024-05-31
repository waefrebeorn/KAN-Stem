import os
import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from multiprocessing import Process, Value, Lock
import logging

from dataset import StemSeparationDataset, collate_fn
from utils import (
    detect_parameters, get_optimizer, custom_loss, write_audio, check_device,
    compute_adversarial_loss, load_and_preprocess, data_augmentation, read_audio
)
from model import KANWithDepthwiseConv, KANDiscriminator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_process = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_processed_stems = Value('i', 0)
total_num_stems = Value('i', 0)
dataset_lock = Lock()

def create_model_and_optimizer(device, n_mels, target_length, cache_dir, learning_rate_g, learning_rate_d, optimizer_name_g, optimizer_name_d, weight_decay):
    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d, weight_decay)
    return model, discriminator, optimizer_g, optimizer_d

def run_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep)

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    global training_process
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]
    training_process = Process(target=run_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"

def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    global device
    logger.info(f"Starting training with dataset at {data_dir}")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256
    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(device, n_mels, target_length, cache_dir, learning_rate_g, learning_rate_d, optimizer_name_g, optimizer_name_d, weight_decay)

    for stem in range(num_stems):
        # Create DataLoaders outside the train_single_stem function
        dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device='cpu' if use_cpu_for_prep else device)
        val_dataset = StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=suppress_warnings, num_workers=num_workers, device='cpu' if use_cpu_for_prep else device)
        
        logger.info("Processing stems for training and validation datasets.")
        dataset._process_stems()
        val_dataset._process_stems()
        
        train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
        
        train_single_stem(stem, train_loader, val_loader, num_epochs, device, checkpoint_dir, save_interval, accumulation_steps, model, discriminator, optimizer_g, optimizer_d, loss_function_g, loss_function_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, early_stopping_patience, weight_decay, suppress_reading_messages)

def train_single_stem(stem, train_loader, val_loader, num_epochs, device, checkpoint_dir, save_interval, accumulation_steps, model, discriminator, optimizer_g, optimizer_d, loss_function_g, loss_function_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, early_stopping_patience, weight_decay, suppress_reading_messages):
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=num_epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler()
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} started.")
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for i, data in enumerate(train_loader):
            if data is None:
                continue

            logger.debug(f"Processing batch {i+1}/{len(train_loader)}")

            inputs = data['input'].unsqueeze(1)
            targets = data['target'][:, stem].unsqueeze(1)

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                loss_g = loss_function_g(outputs, targets)
                scaler.scale(loss_g).backward()

                real_labels = torch.ones(inputs.size(0), 1, device=device)
                fake_labels = torch.zeros(inputs.size(0), 1, device=device)
                real_out = discriminator(targets.clone().detach())
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
            torch.cuda.empty_cache()

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if data is None:
                    continue

                logger.debug(f"Validating batch {i+1}/{len(val_loader)}")

                inputs = data['input'].unsqueeze(1).to(device)
                targets = data['target'][:, stem].unsqueeze(1).to(device)
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]
                loss = loss_function_g(outputs, targets)
                val_loss += loss.item()

                for j in range(outputs.size(0)):
                    sdr, sir, sar = compute_adversarial_loss(targets[j], outputs[j], device)
                    if not torch.isnan(sdr):
                        sdr_total += sdr.mean()
                        sir_total += sir.mean()
                        sar_total += sar.mean()

        num_valid_samples = len(val_loader.dataset) - torch.isnan(sdr_total).sum()
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
