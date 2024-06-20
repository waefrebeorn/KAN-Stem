import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any
from utils import (
    compute_sdr, compute_sir, compute_sar, convert_to_3_channels,
    gradient_penalty, PerceptualLoss, detect_parameters, preprocess_and_cache_dataset,
    StemSeparationDataset, collate_fn, log_training_parameters, ensure_dir_exists,
    get_optimizer, integrated_dynamic_batching, purge_vram, load_from_cache
)
from model_setup import create_model_and_optimizer

logger = logging.getLogger(__name__)

def train_single_stem(
    stem_name: str, 
    dataset: StemSeparationDataset,  
    val_dataset: StemSeparationDataset,
    training_params: dict, 
    model_params: dict,
    sample_rate: int, 
    n_mels: int, 
    n_fft: int, 
    target_length: int, 
    stop_flag: Any,
    suppress_reading_messages: bool = False
):
    if stem_name == "input":
        logger.info(f"Skipping training for stem: {stem_name} (test input)")
        return

    logger.info(f"Starting training for single stem: {stem_name}")
    
    device = torch.device(training_params['device_str'])
    writer = SummaryWriter(log_dir=os.path.join(training_params['checkpoint_dir'], 'runs', f'stem_{stem_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if model_params['tensorboard_flag'] else None

    model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = create_model_and_optimizer(
        training_params['device_str'], n_mels, target_length,
        training_params['initial_lr_g'], training_params['initial_lr_d'], model_params['optimizer_name_g'],
        model_params['optimizer_name_d'], training_params['weight_decay']
    )

    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=10)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)

    early_stopping_counter = 0
    best_val_loss = float('inf')

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

        for i, batch in enumerate(integrated_dynamic_batching(dataset, training_params['batch_size'], stem_name)):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            try:
                # Use the first file path from the batch
                file_path = os.path.join(dataset.cache_dir, batch['file_paths'][0])
                data = load_from_cache(file_path, device)
                inputs, targets = data['input'], data['target']
            except KeyError as e:
                logger.error(f"Batch missing 'file_paths' key: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading data for {file_path}: {e}")
                continue

            if not isinstance(inputs, torch.Tensor):
                logger.error(f"Expected inputs to be a tensor, but got {type(inputs)} instead.")
                continue

            with autocast():
                outputs = model(inputs)
                loss_g = model_params['loss_function_g'](outputs, targets)

                if model_params['perceptual_loss_flag'] and (i % 5 == 0):  # Perceptual loss every 5 steps
                    perceptual_loss = model_params['perceptual_loss_weight'] * PerceptualLoss(feature_extractor)(outputs, targets)
                    loss_g += perceptual_loss

            scaler_g.scale(loss_g).backward()

            if (i + 1) % training_params['discriminator_update_interval'] == 0:
                with autocast():
                    if training_params['add_noise']:
                        noise = torch.randn_like(targets, device=device) * training_params['noise_amount']
                        targets = targets + noise

                    real_out = discriminator(targets)
                    fake_out = discriminator(outputs.detach())

                    loss_d_real = model_params['loss_function_d'](real_out, torch.ones_like(real_out))
                    loss_d_fake = model_params['loss_function_d'](fake_out, torch.zeros_like(fake_out))
                    gp = gradient_penalty(discriminator, targets, outputs.detach(), device)
                    loss_d = (loss_d_real + loss_d_fake) / 2 + gp

                scaler_d.scale(loss_d).backward()

            if (i + 1) % training_params['accumulation_steps'] == 0:
                scaler_g.step(optimizer_g)
                scaler_d.step(optimizer_d)
                scaler_g.update()
                scaler_d.update()
                optimizer_g.zero_grad(set_to_none=True)
                optimizer_d.zero_grad(set_to_none=True)

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item() if 'loss_d' in locals() else 0

            if (i + 1) % 100 == 0:
                purge_vram()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in integrated_dynamic_batching(val_dataset, training_params['batch_size'], stem_name):
                try:
                    file_path = os.path.join(val_dataset.cache_dir, batch['file_paths'][0])
                    data = load_from_cache(file_path, device)
                    inputs, targets = data['input'], data['target']
                except KeyError as e:
                    logger.error(f"Batch missing 'file_paths' key: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error loading data for {file_path}: {e}")
                    continue

                if not isinstance(inputs, torch.Tensor):
                    logger.error(f"Expected inputs to be a tensor, but got {type(inputs)} instead.")
                    continue

                with autocast():
                    outputs = model(inputs)
                    loss = model_params['loss_function_g'](outputs, targets)
                
                val_loss += loss.item()

        if model_params['tensorboard_flag']:
            writer.add_scalar('Loss/Validation', val_loss / len(val_dataset), epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= training_params['early_stopping_patience']:
            logger.info('Early stopping triggered.')
            break

        if (epoch + 1) % training_params['save_interval'] == 0:
            checkpoint_path = os.path.join(training_params['checkpoint_dir'], f'checkpoint_stem_{stem_name}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        purge_vram()

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

    train_dataset, val_dataset = preprocess_and_cache_dataset(
        data_dir, n_mels, target_length, n_fft, training_params['cache_dir'], apply_data_augmentation, training_params['suppress_warnings'], training_params['suppress_reading_messages'], training_params['num_workers'], device, stop_flag
    )

    stem_names = list(train_dataset.stem_names.keys())

    for stem_name in stem_names:
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        if stem_name == "input":
            continue 

        train_single_stem(
            stem_name, train_dataset, val_dataset, training_params, model_params, 
            sample_rate, n_mels, n_fft, target_length, stop_flag, suppress_reading_messages
        )

    logger.info("Training finished.")

if __name__ == '__main__':
    stop_flag = torch.tensor(0, dtype=torch.int32)

    start_training(
        data_dir='path_to_data',
        val_dir='path_to_val_data',
        batch_size=1,  # Train with batch size of one
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
        weight_decay=1e-5,
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
