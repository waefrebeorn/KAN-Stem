import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import (
    compute_sdr, compute_sir, compute_sar,
    gradient_penalty, PerceptualLoss, detect_parameters,
    StemSeparationDataset, collate_fn, log_training_parameters,
    ensure_dir_exists, get_optimizer, purge_vram,
    process_and_cache_dataset, create_dataloader
)
from model_setup import create_model_and_optimizer
import time
import numpy as np  # For creating zero-filled segments

logger = logging.getLogger(__name__)

def validate_tensor_shapes(tensor1, tensor2, message=""):
    if tensor1.shape != tensor2.shape:
        logger.error(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}. {message}")
        raise ValueError(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}. {message}")

def reshape_for_discriminator(x):
    return x.view(x.size(0), -1)

def assemble_full_output(full_outputs: list, skipped_segments: list, target_shape: tuple) -> torch.Tensor:
    """Assembles the full output from segments, filling in skipped segments with zeros."""
    
    full_output = torch.zeros(target_shape)
    current_segment = 0

    for i in range(target_shape[0]):
        if i in skipped_segments:
            # Fill skipped segment with zeros (or appropriate placeholder)
            full_output[i] = torch.zeros_like(full_outputs[0])  
        else:
            # Copy the output of the processed segment
            full_output[i] = full_outputs[current_segment]
            current_segment += 1

    return full_output

def train_single_stem(
    stem_name: str,
    dataset: StemSeparationDataset,
    val_dataset: StemSeparationDataset,
    training_params: dict,
    model_params: dict,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    segment_length: int,
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
        training_params['device_str'], n_mels, segment_length,
        training_params['initial_lr_g'], training_params['initial_lr_d'], model_params['optimizer_name_g'],
        model_params['optimizer_name_d'], training_params['weight_decay']
    )

    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=10)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)

    early_stopping_counter = 0
    best_val_loss = None

    perceptual_loss_fn = PerceptualLoss(sample_rate, n_fft, n_mels).to(device)

    global_step = 0

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

        for i, file_id in enumerate(dataset.file_ids):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            input_file_path = os.path.join(dataset.data_dir, file_id['input_file'])
            target_file_path = os.path.join(dataset.data_dir, file_id['target_files'][stem_name])

            logger.info(f"Input file path: {input_file_path}")
            logger.info(f"Target file path: {target_file_path}")

            if not os.path.exists(input_file_path):
                logger.error(f"Input file does not exist: {input_file_path}")
                continue
            if not os.path.exists(target_file_path):
                logger.error(f"Target file does not exist: {target_file_path}")
                continue

            input_data = dataset.process_and_cache_file(input_file_path, file_id['identifier'], 'input')
            target_data = dataset.process_and_cache_file(target_file_path, file_id['identifier'], stem_name)

            if input_data.numel() == 0 or target_data.numel() == 0:
                logger.warning(f"Skipping empty input or target for file_id: {file_id['identifier']}")
                continue

            full_outputs = []  # Store outputs for the entire track
            all_skipped_segments = []

            for segment_idx in range(input_data.size(0)):
                input_segment = input_data[segment_idx].unsqueeze(0)
                target_segment = target_data[segment_idx].unsqueeze(0)

                logger.info(f"Input tensor shape: {input_segment.shape}")
                logger.info(f"Target tensor shape: {target_segment.shape}")

                with autocast():
                    outputs, skipped_segments = model(input_segment.to(device))
                    logger.info(f"Model output tensor shape: {outputs.shape}")

                    # Ensure target tensor shape matches output tensor shape
                    if target_segment.dim() == 5 and target_segment.size(1) == 1:
                        target_segment = target_segment.squeeze(1)

                    validate_tensor_shapes(outputs, target_segment, "Before computing loss_g")
                    loss_g = model_params['loss_function_g'](outputs, target_segment)

                    if model_params['perceptual_loss_flag'] and (i % 5 == 0):
                        perceptual_loss = model_params['perceptual_loss_weight'] * perceptual_loss_fn(outputs, target_segment)
                        loss_g += perceptual_loss

                scaler_g.scale(loss_g).backward(retain_graph=True)
                purge_vram()

                with autocast():
                    if training_params['add_noise']:
                        noise = torch.randn_like(target_segment, device=device) * training_params['noise_amount']
                        target_segment = target_segment + noise

                    real_out = discriminator(target_segment)  # No need for update_cache here
                    fake_out = discriminator(outputs)  # No need for update_cache here

                    loss_d_real = model_params['loss_function_d'](real_out, torch.ones_like(real_out) * training_params['label_smoothing_real'])
                    loss_d_fake = model_params['loss_function_d'](fake_out, torch.zeros_like(fake_out) * training_params['label_smoothing_fake'])
                    gp = gradient_penalty(discriminator, target_segment, outputs, device)
                    loss_d = (loss_d_real + loss_d_fake) / 2 + gp

                scaler_d.scale(loss_d).backward()
                purge_vram()

                if (i + 1) % training_params['accumulation_steps'] == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad(set_to_none=True)

                running_loss_g += loss_g.item()
                running_loss_d += loss_d.item()

                # Compute and log metrics
                with torch.no_grad():
                    sdr = compute_sdr(target_segment.cpu(), outputs.cpu())
                    sir = compute_sir(target_segment.cpu(), outputs.cpu())
                    sar = compute_sar(target_segment.cpu(), outputs.cpu())

                if model_params['tensorboard_flag']:
                    global_step = epoch * len(dataset.file_ids) * len(input_data) + i * len(input_data) + segment_idx
                    writer.add_scalar(f'Loss/Generator/{stem_name}/Segment', loss_g.item(), global_step)
                    writer.add_scalar(f'Loss/Discriminator/{stem_name}/Segment', loss_d.item(), global_step)
                    writer.add_scalar(f'Metrics/SDR/{stem_name}/Segment', sdr.mean().item(), global_step)
                    writer.add_scalar(f'Metrics/SIR/{stem_name}/Segment', sir.mean().item(), global_step)
                    writer.add_scalar(f'Metrics/SAR/{stem_name}/Segment', sar.mean().item(), global_step)

                if (i + 1) % 100 == 0:
                    if model_params['tensorboard_flag']:
                        writer.add_scalar(f'Loss/Generator/{stem_name}/Batch', running_loss_g / (i + 1), global_step)
                        writer.add_scalar(f'Loss/Discriminator/{stem_name}/Batch', running_loss_d / (i + 1), global_step)
                    logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}] Batch [{i+1}/{len(dataset.file_ids)}]")
                    logger.info(f"Generator Loss: {running_loss_g / (i + 1):.4f}, Discriminator Loss: {running_loss_d / (i + 1):.4f}")
                    purge_vram()

                # Track skipped segments for the entire track
                all_skipped_segments.extend([seg_idx + segment_idx for seg_idx in skipped_segments])

                # Accumulate outputs for the entire track
                full_outputs.append(outputs.detach().cpu())  # Store as detached tensors for efficiency

            # Reassemble full output for validation and metrics calculation
            assembled_output = assemble_full_output(full_outputs, all_skipped_segments, target_data.shape)

        model.eval()
        val_loss = 0.0
        val_sdr = 0.0
        val_sir = 0.0
        val_sar = 0.0
        with torch.no_grad():
            for file_id in val_dataset.file_ids:
                input_file_path = os.path.join(val_dataset.data_dir, file_id['input_file'])
                target_file_path = os.path.join(val_dataset.data_dir, file_id['target_files'][stem_name])

                logger.info(f"Validation input file path: {input_file_path}")
                logger.info(f"Validation target file path: {target_file_path}")

                if not os.path.exists(input_file_path):
                    logger.error(f"Validation input file does not exist: {input_file_path}")
                    continue
                if not os.path.exists(target_file_path):
                    logger.error(f"Validation target file does not exist: {target_file_path}")
                    continue

                input_data = val_dataset.process_and_cache_file(input_file_path, file_id['identifier'], 'input')
                target_data = val_dataset.process_and_cache_file(target_file_path, file_id['identifier'], stem_name)

                if input_data.numel() == 0 or target_data.numel() == 0:
                    logger.warning(f"Skipping empty input or target for file_id: {file_id['identifier']}")
                    continue

                for segment_idx in range(input_data.size(0)):
                    input_segment = input_data[segment_idx].unsqueeze(0)
                    target_segment = target_data[segment_idx].unsqueeze(0)

                    with autocast():
                        outputs, skipped_segments = model(input_segment.to(device))
                        logger.info(f"Validation - Model output tensor shape: {outputs.shape}")

                        # Ensure target tensor shape matches output tensor shape
                        if target_segment.dim() == 5 and target_segment.size(1) == 1:
                            target_segment = target_segment.squeeze(1)

                        validate_tensor_shapes(outputs, target_segment, "During validation before computing loss")
                        loss = model_params['loss_function_g'](outputs, target_segment)
                        val_loss += loss.item()

                        sdr, sir, sar = compute_sdr(target_segment.cpu(), outputs.cpu()), compute_sir(target_segment.cpu(), outputs.cpu()), compute_sar(target_segment.cpu(), outputs.cpu())
                        val_sdr += sdr.mean().item()
                        val_sir += sir.mean().item()
                        val_sar += sar.mean().item()

                        if model_params['tensorboard_flag']:
                            writer.add_scalar(f'Loss/Validation/{stem_name}/Segment', loss.item(), global_step)
                            writer.add_scalar(f'Metrics/SDR_Validation/{stem_name}/Segment', sdr.mean().item(), global_step)
                            writer.add_scalar(f'Metrics/SIR_Validation/{stem_name}/Segment', sir.mean().item(), global_step)
                            writer.add_scalar(f'Metrics/SAR_Validation/{stem_name}/Segment', sar.mean().item(), global_step)

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_sdr = val_sdr / len(val_dataset)
        avg_val_sir = val_sir / len(val_dataset)
        avg_val_sar = val_sar / len(val_dataset)

        if model_params['tensorboard_flag']:
            writer.add_scalar('Loss/Validation/Average', avg_val_loss, epoch + 1)
            writer.add_scalar('Metrics/SDR/Validation_Average', avg_val_sdr, epoch + 1)
            writer.add_scalar('Metrics/SIR/Validation_Average', avg_val_sir, epoch + 1)
            writer.add_scalar('Metrics/SAR/Validation_Average', avg_val_sar, epoch + 1)
            writer.add_scalar('Loss/Generator_Avg', running_loss_g / len(dataset.file_ids), epoch + 1)
            writer.add_scalar('Loss/Discriminator_Avg', running_loss_d / len(dataset.file_ids), epoch + 1)

        logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}]")
        logger.info(f"Generator Loss (Avg): {running_loss_g / len(dataset.file_ids):.4f}")
        logger.info(f"Discriminator Loss (Avg): {running_loss_d / len(dataset.file_ids):.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"SDR: {avg_val_sdr:.4f}, SIR: {avg_val_sir:.4f}, SAR: {avg_val_sar:.4f}")
        logger.info("-----------------------------------")

        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if not training_params['disable_early_stopping'] and early_stopping_counter >= training_params['early_stopping_patience']:
            logger.info('Early stopping triggered.')
            break

        if (epoch + 1) % training_params['save_interval'] == 0:
            checkpoint_path = os.path.join(training_params['checkpoint_dir'], f'stem_{stem_name}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        scheduler_g.step(avg_val_loss)
        scheduler_d.step(avg_val_loss)

    if model_params['tensorboard_flag']:
        writer.close()

    logger.info(f"Training completed for stem: {stem_name}")

def start_training(
    data_dir: str, batch_size: int, num_epochs: int, initial_lr_g: float, initial_lr_d: float,
    use_cuda: bool, checkpoint_dir: str, save_interval: int, accumulation_steps: int, num_stems: int,
    num_workers: int, cache_dir: str, loss_function_g: nn.Module, loss_function_d: nn.Module,
    optimizer_name_g: str, optimizer_name_d: str, perceptual_loss_flag: bool, perceptual_loss_weight: float,
    clip_value: float, scheduler_step_size: int, scheduler_gamma: float, tensorboard_flag: bool,
    add_noise: bool, noise_amount: float, early_stopping_patience: int,
    disable_early_stopping: bool, weight_decay: float, suppress_warnings: bool, suppress_reading_messages: bool,
    discriminator_update_interval: int, label_smoothing_real: float, label_smoothing_fake: float,
    suppress_detailed_logs: bool, stop_flag: torch.Tensor, channel_multiplier: float, segments_per_track: int = 5,
    use_cache: bool = True, update_cache: bool = False
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
        'use_cache': use_cache,
        'update_cache': update_cache
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

    try:
        sample_rate, n_mels, n_fft = detect_parameters(data_dir, cache_dir)
    except ValueError as e:
        logger.error(f"Error detecting parameters: {e}")
        raise

    segment_length = 22050

    dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=segment_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        segments_per_track=segments_per_track
    )

    num_files = len(dataset.file_ids)
    split_index = int(num_files * 0.8)
    train_file_ids = dataset.file_ids[:split_index]
    val_file_ids = dataset.file_ids[split_index:]

    train_dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=segment_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        segments_per_track=segments_per_track,
        file_ids=train_file_ids
    )
    val_dataset = StemSeparationDataset(
        data_dir=data_dir,
        n_mels=n_mels,
        target_length=segment_length,
        n_fft=n_fft,
        cache_dir=cache_dir,
        suppress_reading_messages=suppress_reading_messages,
        device=device,
        segments_per_track=segments_per_track,
        file_ids=val_file_ids
    )

    for stem_name in ['vocals', 'drums', 'bass', 'kick', 'keys', 'guitar']:
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        if stem_name == "input":
            continue

        start_time = time.time()

        train_single_stem(
            stem_name, train_dataset, val_dataset, training_params, model_params,
            sample_rate, n_mels, n_fft, segment_length, stop_flag, suppress_reading_messages
        )

        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Training for stem {stem_name} finished in {epoch_duration:.2f} seconds")

    logger.info("Training finished.")

if __name__ == '__main__':
    stop_flag = torch.multiprocessing.Value('i', 0)

    start_training(
        data_dir='path_to_data',
        batch_size=1,
        num_epochs=10,
        initial_lr_g=1e-4,
        initial_lr_d=1e-4,
        use_cuda=True,
        checkpoint_dir='./checkpoints',
        save_interval=1,
        accumulation_steps=1,
        num_stems=6,
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
        channel_multiplier=2,
        segments_per_track=5,
        use_cache=True,
        update_cache=True
    )
