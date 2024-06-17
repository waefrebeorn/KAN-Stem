import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
import logging
from torchvision import models
from torchvision.models import VGG16_Weights
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from utils import (
    compute_sdr, compute_sir, compute_sar, convert_to_3_channels,
    gradient_penalty, PerceptualLoss, detect_parameters, preprocess_and_cache_dataset,
    StemSeparationDataset, collate_fn, log_training_parameters, ensure_dir_exists,
    get_optimizer, warm_up_cache_batch, monitor_memory_usage, dynamic_batching
)
from model_setup import create_model_and_optimizer
import gc
import numpy as np
from typing import Any, List

logger = logging.getLogger(__name__)

def train_single_stem(
    stem_name: str, dataset: Dataset, val_dataset: Dataset, training_params: dict, model_params: dict,
    sample_rate: int, n_mels: int, n_fft: int, target_length: int, stop_flag: Any,
    suppress_reading_messages: bool = False
):
    if stem_name == "input":
        logger.info(f"Skipping training for stem: {stem_name} (test input)")
        return

    logger.info(f"Starting training for single stem: {stem_name}")
    writer = SummaryWriter(log_dir=os.path.join(training_params['checkpoint_dir'], 'runs', f'stem_{stem_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if model_params['tensorboard_flag'] else None

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(
        training_params['device_str'], n_mels, target_length,
        training_params['initial_lr_g'], training_params['initial_lr_d'], model_params['optimizer_name_g'],
        model_params['optimizer_name_d'], training_params['weight_decay'], model_params['channel_multiplier']
    )

    feature_extractor = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(training_params['device_str']).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    def choose_scheduler(optimizer, optimizer_name):
        if optimizer_name in ['SGD', 'Momentum', 'RMSprop']:
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

    for epoch in range(training_params['num_epochs']):
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        logger.info(f"Epoch {epoch+1}/{training_params['num_epochs']} started.")
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0

        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        for i, (inputs, targets, file_paths) in enumerate(dynamic_batching(dataset, training_params['batch_size'])):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            if inputs is None or targets is None:
                continue

            logger.debug(f"Processing batch {i+1}/{len(dataset)}")

            inputs = inputs.to(training_params['device_str']).float()
            targets = targets.to(training_params['device_str']).float()

            input_segments = inputs.chunk(training_params.get('segments_per_track', 1), dim=-1)
            target_segments = targets.chunk(training_params.get('segments_per_track', 1), dim=-1)

            with autocast():
                for input_seg, target_seg in zip(input_segments, target_segments):
                    outputs = model(input_seg)

                    if outputs is None:
                        logger.error(f"Model returned None for batch {i+1}. Skipping this batch.")
                        continue

                    loss_g = model_params['loss_function_g'](outputs, target_seg)

                    if model_params['perceptual_loss_flag'] and (i % perceptual_loss_frequency == 0):
                        outputs_3ch = convert_to_3_channels(outputs.cpu())
                        targets_3ch = convert_to_3_channels(target_seg.cpu())
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
                            noise = torch.randn_like(target_seg) * training_params['noise_amount']
                            target_seg = target_seg + noise

                        real_labels = torch.full((input_seg.size(0), 1), training_params['label_smoothing_real'], device=training_params['device_str'], dtype=torch.float)
                        fake_labels = torch.full((input_seg.size(0), 1), training_params['label_smoothing_fake'], device=training_params['device_str'], dtype=torch.float)
                        real_out = discriminator(target_seg.clone().detach())
                        fake_out = discriminator(outputs.clone().detach())

                        if real_out is None or fake_out is None:
                            logger.error(f"Discriminator outputs or labels are None. Skipping batch {i+1}.")
                            continue

                        loss_d_real = model_params['loss_function_d'](real_out, real_labels)
                        loss_d_fake = model_params['loss_function_d'](fake_out, fake_labels)
                        gp = gradient_penalty(discriminator, target_seg, outputs, training_params['device_str'])
                        loss_d = (loss_d_real + loss_d_fake) / 2 + gp

                        running_loss_d += loss_d.item() / accumulation_steps
                        scaler.scale(loss_d).backward()
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), model_params['clip_value'])
                        scaler.step(optimizer_d)
                        scaler.update()
                        optimizer_d.zero_grad(set_to_none=True)

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['clip_value'])
                scaler.step(optimizer_g)
                scaler.update()
                optimizer_g.zero_grad(set_to_none=True)

                logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}], Step [{i+1}/{len(dataset)}], Loss G: {running_loss_g:.4f}, Loss D: {running_loss_d:.4f}")
                if model_params['tensorboard_flag']:
                    writer.add_scalar('Loss/Generator', running_loss_g, epoch * len(dataset) + i)
                    writer.add_scalar('Loss/Discriminator', running_loss_d, epoch * len(dataset) + i)

                if isinstance(scheduler_g, CyclicLR):
                    scheduler_g.step()
                    scheduler_d.step()

            torch.cuda.empty_cache()

        if isinstance(scheduler_g, ReduceLROnPlateau):
            scheduler_g.step(running_loss_g / len(dataset))
            scheduler_d.step(running_loss_d / len(dataset))

        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        num_sdr_samples, num_sir_samples, num_sar_samples = 0, 0, 0

        with torch.no_grad():
            try:
                for i, (inputs, targets, file_paths) in enumerate(dynamic_batching(val_dataset, training_params['batch_size'])):
                    if stop_flag.value == 1:
                        logger.info("Training stopped.")
                        return

                    if inputs is None or targets is None:
                        continue

                    logger.debug(f"Validating batch {i+1}/{len(val_dataset)}")

                    inputs = inputs.to(training_params['device_str']).float()
                    targets = targets.to(training_params['device_str']).float()
                    outputs = model(inputs)

                    if outputs is None:
                        logger.error(f"Model returned None for validation batch {i+1}. Skipping this batch.")
                        continue

                    logger.debug(f"Validation output shape: {outputs.shape}")

                    loss = model_params['loss_function_g'](outputs, targets)
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

                val_loss /= len(val_dataset)
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
            checkpoint_path = os.path.join(training_params['checkpoint_dir'], f'checkpoint_stem_{stem_name}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        monitor_memory_usage()

    final_model_path = f"{training_params['checkpoint_dir']}/model_final_stem_{stem_name}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem_name}. Final model saved at {final_model_path}")

    if model_params['tensorboard_flag']:
        writer.close()
        
def start_training(
    data_dir: str, val_dir: str, batch_size: int, num_epochs: int, initial_lr_g: float, initial_lr_d: float, 
    use_cuda: bool, checkpoint_dir: str, save_interval: int, accumulation_steps: int, num_stems: int, 
    num_workers: int, cache_dir: str, loss_function_g: nn.Module, loss_function_d: nn.Module, 
    optimizer_name_g: str, optimizer_name_d: str, perceptual_loss_flag: bool, perceptual_loss_weight: float, 
    clip_value: float, scheduler_step_size: int, scheduler_gamma: float, tensorboard_flag: bool, 
    apply_data_augmentation: bool, add_noise: bool, noise_amount: float, early_stopping_patience: int, 
    disable_early_stopping: bool, weight_decay: float, suppress_warnings: bool, suppress_reading_messages: bool, 
    discriminator_update_interval: int, label_smoothing_real: float, label_smoothing_fake: float, 
    suppress_detailed_logs: bool, stop_flag: Any, use_cache: bool, channel_multiplier: float, segments_per_track: int = 10
):
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
        'discriminator_update_interval': discriminator_update_interval,
        'label_smoothing_real': label_smoothing_real,
        'label_smoothing_fake': label_smoothing_fake,
        'suppress_detailed_logs': suppress_detailed_logs,
        'segments_per_track': segments_per_track,
        'use_cache': use_cache
    }
    model_params = {
        'optimizer_name_g': optimizer_name_g,
        'optimizer_name_d': optimizer_name_d,
        'loss_function_g': loss_function_g,
        'loss_function_d': loss_function_d,
        'perceptual_loss_flag': perceptual_loss_flag,
        'perceptual_loss_weight': perceptual_loss_weight,
        'clip_value': clip_value,
        'tensorboard_flag': tensorboard_flag,
        'channel_multiplier': channel_multiplier
    }

    log_training_parameters(training_params)

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = sample_rate // 2  # Assuming a 0.5-second target length

    if use_cache:
        dataset, mel_spectrogram = preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], False, training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device, stop_flag)
        if apply_data_augmentation:
            val_dataset, _ = preprocess_and_cache_dataset(val_dir, n_mels, target_length, n_fft, training_params['cache_dir'], False, training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device, stop_flag)
        else:
            val_dataset = dataset  # Use the same dataset for training and validation
    else:
        raise ValueError("use_cache must be True for this code.")

    for stem_name in dataset.stem_names.keys():
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        train_single_stem(stem_name, dataset, val_dataset, training_params, model_params, sample_rate, n_mels, n_fft, target_length, stop_flag)

    logger.info("Training finished.")

if __name__ == '__main__':
    import multiprocessing
    stop_flag = multiprocessing.Value('i', 0)
    
    start_training(
        data_dir='path_to_data',
        val_dir='path_to_val_data',
        batch_size=32,
        num_epochs=10,
        initial_lr_g=1e-4,
        initial_lr_d=1e-4,
        use_cuda=True,
        checkpoint_dir='./checkpoints',
        save_interval=1,
        accumulation_steps=1,
        num_stems=4,
        num_workers=4,
        cache_dir='./cache',
        loss_function_g=torch.nn.L1Loss(),
        loss_function_d=torch.nn.BCELoss(),
        optimizer_name_g='Adam',
        optimizer_name_d='Adam',
        perceptual_loss_flag=True,
        perceptual_loss_weight=0.1,
        clip_value=0.1,
        scheduler_step_size=10,
        scheduler_gamma=0.1,
        tensorboard_flag=True,
        apply_data_augmentation=False,
        add_noise=False,
        noise_amount=0.1,
        early_stopping_patience=5,
        disable_early_stopping=False,
        weight_decay=0,
        suppress_warnings=False,
        suppress_reading_messages=False,
        discriminator_update_interval=5,
        label_smoothing_real=0.9,
        label_smoothing_fake=0.1,
        suppress_detailed_logs=False,
        stop_flag=stop_flag,
        use_cache=True,
        channel_multiplier=0.5,
        segments_per_track=10  # Adjust this value as needed
    )
