import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import logging
from torchvision import models
from torchvision.models import VGG16_Weights
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from dataset import StemSeparationDataset, collate_fn
from utils import compute_sdr, compute_sir, compute_sar, convert_to_3_channels, gradient_penalty, PerceptualLoss, detect_parameters
from data_preprocessing import preprocess_and_cache_dataset
from model_setup import create_model_and_optimizer
from functools import lru_cache
import gc
from cachetools import LRUCache, cached
import numpy as np
import h5py

logger = logging.getLogger(__name__)

def log_training_parameters(params):
    """Logs the training parameters."""
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

@cached(cache=LRUCache(maxsize=100))
def preprocess_data(input_data, target_data, n_mels, segment_size, n_fft, device):
    """Preprocess a single batch or segment on the CPU."""
    inputs = input_data.to('cpu')
    targets = target_data.to('cpu')
    # Apply preprocessing steps if any
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets

def monitor_memory_usage(training_params):
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(training_params['device_str']) / 1024 ** 3
        logger.info(f'GPU memory allocated: {gpu_memory:.2f} GB')
    else:
        logger.info('GPU not available, skipping memory monitoring.')

def dynamic_cache_management(preprocess_and_cache):
    # Monitor GPU memory usage and adjust cache size dynamically
    if torch.cuda.is_available():
        reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3
        max_memory = torch.cuda.max_memory_reserved() / 1024 ** 3
        if reserved_memory < max_memory * 0.7:
            current_cache_size = preprocess_and_cache.cache_info().currsize
            new_cache_size = min(current_cache_size + 10, 100)
            preprocess_and_cache.cache_clear()
        elif reserved_memory > max_memory * 0.9:
            current_cache_size = preprocess_and_cache.cache_info().currsize
            new_cache_size = max(current_cache_size - 10, 10)
            preprocess_and_cache.cache_clear()
        preprocess_and_cache = cached(cache=LRUCache(maxsize=new_cache_size))(preprocess_and_cache.cache_info().func)
        logger.info(f'Cache size adjusted to: {preprocess_and_cache.cache_info().maxsize}')

def warm_up_cache(loader, preprocess_data, n_mels, target_length, n_fft, device_str, num_batches=5):
    for i, data in enumerate(loader):
        _ = preprocess_data(data['input'], data['target'], n_mels, target_length, n_fft, device_str)
        if i >= num_batches:
            break

def save_batch_to_hdf5(data, batch_idx, dataset_type, cache_dir='./cache'):
    file_path = os.path.join(cache_dir, f"{dataset_type}_batch_{batch_idx}.h5")
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_type, data=data.cpu().numpy())

def load_batch_from_hdf5(batch_idx, dataset_type, cache_dir='./cache'):
    file_path = os.path.join(cache_dir, f"{dataset_type}_batch_{batch_idx}.h5")
    try:
        with h5py.File(file_path, 'r') as f:
            return torch.tensor(f[dataset_type][:])
    except FileNotFoundError:
        return None

def dynamic_max_split_size():
    if torch.cuda.is_available():
        free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        free_memory_mb = free_memory / 1024 / 1024
        max_split_size_mb = max(32, int(free_memory_mb * 0.1))  # Use 10% of free memory, with a minimum of 32 MB
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,max_split_size_mb:{max_split_size_mb}'
        logger.info(f'Set PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True,max_split_size_mb:{max_split_size_mb}')

def train_single_stem(stem, dataset, val_dir, training_params, model_params, sample_rate, n_mels, n_fft, target_length, stop_flag, suppress_reading_messages=False):
    dynamic_max_split_size()  # Set the environment variable dynamically
    
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(training_params['checkpoint_dir'], 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if model_params['tensorboard_flag'] else None

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(
        training_params['device_str'], n_mels, target_length, training_params['cache_dir'],
        training_params['initial_lr_g'], training_params['initial_lr_d'], model_params['optimizer_name_g'],
        model_params['optimizer_name_d'], training_params['weight_decay']
    )

    feature_extractor = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(training_params['device_str']).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    prep_device = 'cpu' if training_params['use_cpu_for_prep'] else training_params['device_str']

    train_loader = DataLoader(
        dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=training_params['num_workers'],
        collate_fn=collate_fn
    )

    val_dataset = StemSeparationDataset(
        val_dir, n_mels, target_length, n_fft, training_params['cache_dir'], apply_data_augmentation=False, 
        suppress_warnings=training_params['suppress_warnings'], suppress_reading_messages=training_params['suppress_reading_messages'], 
        num_workers=training_params['num_workers'], device_prep=prep_device, stop_flag=stop_flag, use_cache=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=training_params['num_workers'],
        collate_fn=collate_fn
    )

    def choose_scheduler(optimizer, optimizer_name):
        if optimizer_name in ['SGD', 'Momentum', 'RMSProp']:
            return CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2')
        else:
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    scheduler_g = choose_scheduler(optimizer_g, model_params['optimizer_name_g'])
    scheduler_d = choose_scheduler(optimizer_d, model_params['optimizer_name_d'])

    scaler = GradScaler()
    early_stopping_counter = 0
    best_val_loss = float('inf')

    perceptual_loss_frequency = 5  # Calculate perceptual loss every 5 mini-batches

    accumulation_steps = training_params['accumulation_steps']
    
    # Cache warming
    warm_up_cache(train_loader, preprocess_data, n_mels, target_length, n_fft, training_params['device_str'])
    warm_up_cache(val_loader, preprocess_data, n_mels, target_length, n_fft, training_params['device_str'])

    for epoch in range(training_params['num_epochs']):
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        logger.info(f"Epoch {epoch+1}/{training_params['num_epochs']} started.")
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for i, data in enumerate(train_loader):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            if data is None:
                continue

            logger.debug(f"Processing batch {i+1}/{len(train_loader)}")

            inputs_segments = torch.split(data['input'], target_length // 10, dim=-1)
            target_segments = torch.split(data['target'], target_length // 10, dim=-1)

            # Pad last segments if needed
            if inputs_segments[-1].shape[-1] != target_length // 10:
                padding_needed = target_length // 10 - inputs_segments[-1].shape[-1]
                inputs_segments[-1] = F.pad(inputs_segments[-1], (0, padding_needed))

            if target_segments[-1].shape[-1] != target_length // 10:
                padding_needed = target_length // 10 - target_segments[-1].shape[-1]
                target_segments[-1] = F.pad(target_segments[-1], (0, padding_needed))

            combined_inputs = []
            combined_targets = []
            combined_outputs = []

            for j, (inputs, targets) in enumerate(zip(inputs_segments, target_segments)):
                inputs, targets = preprocess_data(inputs, targets, n_mels, target_length // 10, n_fft, training_params['device_str'])

                logger.debug(f"Input shape before reshaping: {inputs.shape}")

                if inputs.dim() == 2:
                    inputs = inputs.view(inputs.size(0), 1, n_mels, target_length // 10)
                elif inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                elif inputs.dim() == 4 and inputs.size(1) != 1:
                    inputs = inputs.view(inputs.size(0), 1, n_mels, target_length // 10)

                logger.debug(f"Input shape after reshaping: {inputs.shape}")

                with autocast():
                    outputs = model(inputs)
                    outputs = outputs[..., :target_length // 10]

                    if targets.dim() == 2:
                        targets = targets.view(targets.size(0), 1, n_mels, target_length // 10)
                    if targets.dim() == 3:
                        targets = targets.unsqueeze(1)
                    if outputs.dim() == 2:
                        outputs = outputs.view(outputs.size(0), 1, n_mels, target_length // 10)
                    if outputs.dim() == 3:
                        outputs = outputs.view(outputs.size(0), 1, n_mels, target_length // 10)

                    logger.debug(f"Output shape before squeezing: {outputs.shape}")
                    logger.debug(f"Target shape before squeezing: {targets.shape}")

                    if outputs.dim() == 5:
                        outputs = outputs.squeeze(2)
                    if targets.dim() == 5:
                        targets = targets.squeeze(2)

                    logger.debug(f"Output shape after squeezing: {outputs.shape}")
                    logger.debug(f"Target shape after squeezing: {targets.shape}")

                    combined_inputs.append(inputs.detach().cpu())
                    combined_targets.append(targets.detach().cpu())
                    combined_outputs.append(outputs.detach().cpu())

                    loss_g = model_params['loss_function_g'](outputs.to(training_params['device_str']), targets.to(training_params['device_str']))

                    if model_params['perceptual_loss_flag'] and (i % perceptual_loss_frequency == 0):
                        outputs_3ch = convert_to_3_channels(outputs.cpu())
                        targets_3ch = convert_to_3_channels(targets.cpu())
                        perceptual_loss = model_params['perceptual_loss_weight'] * PerceptualLoss(feature_extractor)(
                            outputs_3ch.to(training_params['device_str']), targets_3ch.to(training_params['device_str'])
                        )
                        loss_g += perceptual_loss.to(training_params['device_str'])

                        del outputs_3ch, targets_3ch
                        gc.collect()
                        torch.cuda.empty_cache()

                    running_loss_g += loss_g.item() / accumulation_steps

                    scaler.scale(loss_g).backward(retain_graph=True)

                    if (i + 1) % training_params['discriminator_update_interval'] == 0:
                        if training_params['add_noise']:
                            noise = torch.randn_like(targets) * training_params['noise_amount']
                            targets = targets + noise

                        real_labels = torch.full((inputs.size(0), 1), training_params['label_smoothing_real'], device=training_params['device_str'], dtype=torch.float)
                        fake_labels = torch.full((inputs.size(0), 1), training_params['label_smoothing_fake'], device=training_params['device_str'], dtype=torch.float)
                        real_out = discriminator(targets.to(training_params['device_str']).clone().detach())
                        fake_out = discriminator(outputs.clone().detach())

                        if real_out is None or fake_out is None:
                            logger.error(f"Discriminator outputs or labels are None. Skipping batch {i+1}.")
                            continue

                        loss_d_real = model_params['loss_function_d'](real_out, real_labels)
                        loss_d_fake = model_params['loss_function_d'](fake_out, fake_labels)
                        gp = gradient_penalty(discriminator, targets.to(training_params['device_str']), outputs.to(training_params['device_str']), training_params['device_str'])
                        loss_d = (loss_d_real + loss_d_fake) / 2 + gp

                        running_loss_d += loss_d.item() / accumulation_steps

                        scaler.scale(loss_d).backward()
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), model_params['clip_value'])

                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['clip_value'])
                        scaler.step(optimizer_g)
                        optimizer_g.zero_grad()
                        scaler.update()

                        scaler.step(optimizer_d)
                        optimizer_d.zero_grad()
                        scaler.update()

                        logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g:.4f}, Loss D: {running_loss_d:.4f}")
                        if model_params['tensorboard_flag']:
                            writer.add_scalar('Loss/Generator', running_loss_g, epoch * len(train_loader) + i)
                            writer.add_scalar('Loss/Discriminator', running_loss_d, epoch * len(train_loader) + i)

                    if isinstance(scheduler_g, CyclicLR):
                        optimizer_g.step()
                        scheduler_g.step()
                        optimizer_d.step()
                        scheduler_d.step()

            # Combine all segments into one tensor and save to HDF5 cache
            combined_inputs = torch.cat(combined_inputs, dim=-1)
            combined_targets = torch.cat(combined_targets, dim=-1)
            combined_outputs = torch.cat(combined_outputs, dim=-1)

            save_batch_to_hdf5(combined_inputs, f'inputs_epoch{epoch}_batch{i}', 'inputs', training_params['cache_dir'])
            save_batch_to_hdf5(combined_targets, f'targets_epoch{epoch}_batch{i}', 'targets', training_params['cache_dir'])
            save_batch_to_hdf5(combined_outputs, f'outputs_epoch{epoch}_batch{i}', 'outputs', training_params['cache_dir'])

            # Clear cache frequently
            preprocess_data.cache_clear()

        if isinstance(scheduler_g, ReduceLROnPlateau):
            optimizer_g.step()
            scheduler_g.step(running_loss_g / len(train_loader))
            optimizer_d.step()
            scheduler_d.step(running_loss_d / len(train_loader))

        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        num_sdr_samples, num_sir_samples, num_sar_samples = 0, 0, 0

        with torch.no_grad():
            try:
                for i, data in enumerate(val_loader):
                    if stop_flag.value == 1:
                        logger.info("Training stopped.")
                        return

                    if data is None:
                        continue

                    logger.debug(f"Validating batch {i+1}/{len(val_loader)}")

                    inputs, targets = preprocess_data(data['input'], data['target'], n_mels, target_length, n_fft, training_params['device_str'])

                    logger.debug(f"Validation input shape before reshaping: {inputs.shape}")

                    if inputs.dim() == 2:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)
                    elif inputs.dim() == 3:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)
                    elif inputs.dim() == 4 and inputs.size(1) != 1:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)

                    logger.debug(f"Validation input shape after reshaping: {inputs.shape}")

                    outputs = model(inputs)
                    outputs = outputs[..., :target_length]

                    if targets.dim() == 2:
                        targets = targets.view(targets.size(0), 1, n_mels, target_length)
                    if targets.dim() == 3:
                        targets = targets.view(targets.size(0), 1, n_mels, target_length)
                    if outputs.dim() == 2:
                        outputs = outputs.view(outputs.size(0), 1, n_mels, target_length)
                    if outputs.dim() == 3:
                        outputs = outputs.view(outputs.size(0), 1, n_mels, target_length)

                    if outputs.dim() == 5:
                        outputs = outputs.squeeze(2)
                    if targets.dim() == 5:
                        targets = targets.squeeze(2)

                    loss = model_params['loss_function_g'](outputs.to(training_params['device_str']), targets.to(training_params['device_str']))
                    val_loss += loss.item()

                    sdr = compute_sdr(targets, outputs)
                    sir = compute_sir(targets, outputs)
                    sar = compute_sar(targets, outputs)

                    sdr_total += torch.nan_to_num(sdr, nan=0.0).sum()
                    sir_total += torch.nan_to_num(sir, nan=0.0).sum()
                    sar_total += torch.nan_to_num(sar, nan=0.0).sum()

                    num_sdr_samples += torch.isfinite(sdr).sum().item()
                    num_sir_samples += torch.isfinite(sir).sum().item()
                    num_sar_samples += torch.isfinite(sar).sum().item()

                val_loss /= len(val_loader)
                sdr_avg = sdr_total / max(num_sdr_samples, 1)
                sir_avg = sir_total / max(num_sir_samples, 1)
                sar_avg = sar_total / max(num_sar_samples, 1)

                logger.info(f'Validation Loss: {val_loss:.4f}, SDR: {sdr_avg:.4f}, SIR: {sir_avg:.4f}, SAR: {sar_avg:.4f}')
                if model_params['tensorboard_flag']:
                    writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
                    writer.add_scalar('Metrics/SDR', sdr_avg, epoch + 1)
                    writer.add_scalar('Metrics/SIR', sir_avg, epoch + 1)
                    writer.add_scalar('Metrics/SAR', sar_avg, epoch + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if not training_params['disable_early_stopping'] and early_stopping_counter >= training_params['early_stopping_patience']:
                    logger.info('Early stopping triggered.')
                    break
            except Exception as e:
                logger.error(f"Error during validation step: {e}", exc_info=True)

        if (epoch + 1) % training_params['save_interval'] == 0:
            checkpoint_path = os.path.join(training_params['checkpoint_dir'], f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        preprocess_data.cache_clear()
        extract_and_cache_features.cache_clear()

        monitor_memory_usage(training_params)

    final_model_path = f"{training_params['checkpoint_dir']}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if model_params['tensorboard_flag']:
        writer.close()

def start_training(data_dir, val_dir, batch_size, num_epochs, initial_lr_g, initial_lr_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, stop_flag, use_cache):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    training_params = {
        'device_str': str(device),
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'initial_lr_g': initial_lr_g,
        'initial_lr_d': initial_lr_d,
        'checkpoint_dir': checkpoint_dir,
        'save_interval': save_interval,
        'accumulation_steps': accumulation_steps,
        'num_workers': num_workers,
        'cache_dir': cache_dir,
        'scheduler_step_size': scheduler_step_size,
        'scheduler_gamma': scheduler_gamma,
        'tensorboard_flag': tensorboard_flag,
        'apply_data_augmentation': apply_data_augmentation,
        'add_noise': add_noise,
        'noise_amount': noise_amount,
        'early_stopping_patience': early_stopping_patience,
        'disable_early_stopping': disable_early_stopping,
        'weight_decay': weight_decay,
        'suppress_warnings': suppress_warnings,
        'suppress_reading_messages': suppress_reading_messages,
        'use_cpu_for_prep': use_cpu_for_prep,
        'discriminator_update_interval': discriminator_update_interval,
        'label_smoothing_real': label_smoothing_real,
        'label_smoothing_fake': label_smoothing_fake,
        'suppress_detailed_logs': suppress_detailed_logs
    }
    model_params = {
        'optimizer_name_g': optimizer_name_g,
        'optimizer_name_d': optimizer_name_d,
        'loss_function_g': loss_function_g,
        'loss_function_d': loss_function_d,
        'perceptual_loss_flag': perceptual_loss_flag,
        'perceptual_loss_weight': perceptual_loss_weight,
        'clip_value': clip_value,
        'tensorboard_flag': tensorboard_flag
    }

    log_training_parameters(training_params)

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = sample_rate // 2  # Assuming a 0.5-second target length

    if use_cache:
        preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], False, training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device, stop_flag)
        if apply_data_augmentation:
            preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], True, training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device, stop_flag)
        
        dataset = StemSeparationDataset(
            data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], apply_data_augmentation=training_params['apply_data_augmentation'],
            suppress_warnings=training_params['suppress_warnings'], suppress_reading_messages=training_params['suppress_reading_messages'],
            num_workers=training_params['num_workers'], device_prep=device, stop_flag=stop_flag, use_cache=use_cache
        )
    else:
        raise ValueError("use_cache must be True for this code.")

    for stem in range(num_stems):
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        train_single_stem(stem, dataset, val_dir, training_params, model_params, sample_rate, n_mels, n_fft, target_length, stop_flag)

    logger.info("Training finished.")
