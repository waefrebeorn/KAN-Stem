import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any
from torchvision.models import vgg16, VGG16_Weights
from utils import (
    compute_sdr, compute_sir, compute_sar,
    gradient_penalty, PerceptualLoss, detect_parameters, 
    StemSeparationDataset, collate_fn, log_training_parameters, 
    ensure_dir_exists, get_optimizer, integrated_dynamic_batching, purge_vram, load_from_cache,
    process_and_cache_dataset  
)
from model_setup import create_model_and_optimizer
import time

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
    stop_flag: torch.Tensor,
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

    feature_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

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

        for i, batch in enumerate(integrated_dynamic_batching(dataset.data_dir, dataset.cache_dir, dataset.n_mels, dataset.n_fft, dataset.target_length, training_params['batch_size'], device, stem_name, dataset.file_ids, shuffle=False)):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            try:
                identifier = batch['file_paths'][0]
                input_cache_file_name = f"input_{identifier}_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
                target_cache_file_name = f"{stem_name}_{identifier}_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
                input_cache_file_path = os.path.join(dataset.cache_dir, input_cache_file_name)
                target_cache_file_path = os.path.join(dataset.cache_dir, target_cache_file_name)

                if not os.path.exists(input_cache_file_path) or not os.path.exists(target_cache_file_path):
                    logger.warning(f"Cache files not found for {identifier}. Processing now.")
                    input_file_path = os.path.join(dataset.data_dir, f'input_{identifier}.wav')
                    dataset.process_and_cache_file(input_file_path, identifier, "input")

                    target_file_path = os.path.join(dataset.data_dir, f'{stem_name}_{identifier}.wav')
                    dataset.process_and_cache_file(target_file_path, identifier, stem_name)

                data = load_from_cache(input_cache_file_path, device)
                inputs, targets = data['input'], load_from_cache(target_cache_file_path, device)['input']
            except KeyError as e:
                logger.error(f"Batch missing 'file_paths' key: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading data for {identifier}: {e}")
                continue

            if not isinstance(inputs, torch.Tensor):
                logger.error(f"Expected inputs to be a tensor, but got {type(inputs)} instead.")
                continue

            with autocast():
                outputs = model(inputs)
                loss_g = model_params['loss_function_g'](outputs, targets)

                if model_params['perceptual_loss_flag'] and (i % 5 == 0):
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
            for batch in integrated_dynamic_batching(val_dataset.data_dir, val_dataset.cache_dir, val_dataset.n_mels, val_dataset.n_fft, val_dataset.target_length, training_params['batch_size'], device, stem_name, val_dataset.file_ids, shuffle=False):
                try:
                    identifier = batch['file_paths'][0]
                    input_cache_file_name = f"input_{identifier}_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
                    target_cache_file_name = f"{stem_name}_{identifier}_{dataset.n_mels}_{dataset.target_length}_{dataset.n_fft}.h5"
                    input_cache_file_path = os.path.join(dataset.cache_dir, input_cache_file_name)
                    target_cache_file_path = os.path.join(dataset.cache_dir, target_cache_file_name)

                    if not os.path.exists(input_cache_file_path) or not os.path.exists(target_cache_file_path):
                        logger.warning(f"Cache files not found for {identifier}. Processing now.")
                        input_file_path = os.path.join(dataset.data_dir, f'input_{identifier}.wav')
                        dataset.process_and_cache_file(input_file_path, identifier, "input")

                        target_file_path = os.path.join(dataset.data_dir, f'{stem_name}_{identifier}.wav')
                        dataset.process_and_cache_file(target_file_path, identifier, stem_name)

                    data = load_from_cache(input_cache_file_path, device)
                    inputs, targets = data['input'], load_from_cache(target_cache_file_path, device)['input']
                except KeyError as e:
                    logger.error(f"Batch missing 'file_paths' key: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error loading data for {identifier}: {e}")
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
    add_noise: bool, noise_amount: float, early_stopping_patience: int, 
    disable_early_stopping: bool, weight_decay: float, suppress_warnings: bool, suppress_reading_messages: bool, 
    discriminator_update_interval: int, label_smoothing_real: float, label_smoothing_fake: float, 
    suppress_detailed_logs: bool, stop_flag: torch.Tensor, use_cache: bool, channel_multiplier: float, segments_per_track: int = 10
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
    target_length = sample_rate // 2

    train_file_ids, val_file_ids = process_and_cache_dataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        n_mels=n_mels,
        n_fft=n_fft,
        device=device,
        suppress_reading_messages=suppress_reading_messages
    )

    train_dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=target_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        use_cache=True,
        segments_per_track=segments_per_track
    )
    val_dataset = StemSeparationDataset(
        data_dir=val_dir,
        n_mels=n_mels,
        target_length=target_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        use_cache=True,
        segments_per_track=segments_per_track
    )

    for stem_name in ['vocals', 'drums', 'bass', 'other']:
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        if stem_name == "input":
            continue 

        start_time = time.time()

        train_single_stem(
            stem_name, train_dataset, val_dataset, training_params, model_params, 
            sample_rate, n_mels, n_fft, target_length, stop_flag, suppress_reading_messages
        )

        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Training for stem {stem_name} finished in {epoch_duration:.2f} seconds")

    logger.info("Training finished.")

if __name__ == '__main__':
    stop_flag = torch.multiprocessing.Value('i', 0)

    start_training(
        data_dir='path_to_data',
        val_dir='path_to_val_data',
        batch_size=1,
        num_epochs=10,
        initial_lr_g=1e-4,
        initial_lr_d=1e-4,
        use_cuda=True,
        checkpoint_dir='./checkpoints',
        save_interval=1,
        accumulation_steps=1,
        num_stems=4,
        num_workers=1,
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
        add_noise=False,
        noise_amount=0.1,
        early_stopping_patience=5,
        disable_early_stopping=False,
        weight_decay=1e-5,
        suppress_warnings=False,
        suppress_reading_messages=True,
        discriminator_update_interval=1,
        label_smoothing_real=0.9,
        label_smoothing_fake=0.1,
        suppress_detailed_logs=False,
        stop_flag=stop_flag,
        use_cache=True,
        channel_multiplier=2,
        segments_per_track=10
    )
