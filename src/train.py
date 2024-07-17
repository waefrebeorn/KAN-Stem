import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from multiprocessing import Value, Process, Manager
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import (
    compute_sdr, compute_sir, compute_sar,
    gradient_penalty, PerceptualLoss, detect_parameters,
    StemSeparationDataset, collate_fn, log_training_parameters,
    ensure_dir_exists, get_optimizer, purge_vram,
    process_and_cache_dataset, create_dataloader
)
from model_setup import create_model_and_optimizer, initialize_model
from utils_checkpoint import save_checkpoint, load_checkpoint
import time
import numpy as np  # For creating zero-filled segments
from torch.multiprocessing import Manager, Process, Value
from multiprocessing.managers import DictProxy
from loss_functions import wasserstein_loss

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
    stop_flag: Value,
    checkpoint_flag: Value,
    training_state: dict,
    current_segment: int,
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

    # Ensure training_state is a dictionary-like object
    training_state = dict(training_state) if isinstance(training_state, DictProxy) else training_state

    # Store in training_state
    training_state.update({
        "model": model,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "scaler_g": scaler_g,
        "scaler_d": scaler_d,
        "stem_name": stem_name,
        "model_params": model_params,
        "training_params": training_params,
        "training_started": True,  # Indicate training has started
    })

    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=10)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=10)

    early_stopping_counter = 0
    best_val_loss = None

    perceptual_loss_fn = PerceptualLoss(sample_rate, n_fft, n_mels).to(device)

    global_step = 0

    # Calculate the total segments in the dataset
    total_segments = sum(
        dataset.process_and_cache_file(os.path.join(dataset.data_dir, file_id['input_file']), file_id['identifier'], 'input').size(0)
        for file_id in dataset.file_ids
    )

    segment_count = current_segment

    for epoch in range(training_state.get('current_epoch', 0), training_params['num_epochs']):
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

        training_state['current_epoch'] = epoch  # Update current epoch in training_state

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

            for segment_idx in range(segment_count, input_data.size(0)):
                input_segment = input_data[segment_idx].unsqueeze(0)
                target_segment = target_data[segment_idx].unsqueeze(0)

                if model_params['tensorboard_flag']:
                    logger.debug(f"Input tensor shape: {input_segment.shape}")
                    logger.debug(f"Target tensor shape: {target_segment.shape}")

                with autocast():
                    outputs = model(input_segment.to(device))
                    if model_params['tensorboard_flag']:
                        logger.debug(f"Model output tensor shape: {outputs.shape}")

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

                segment_count += 1
                training_state['current_segment'] = segment_count  # Update current segment

                if model_params['tensorboard_flag']:
                    global_step = epoch * len(dataset.file_ids) * len(input_data) + i * len(input_data) + segment_idx
                    writer.add_scalar(f'Loss/Generator/{stem_name}/Segment', loss_g.item(), global_step)
                    writer.add_scalar(f'Loss/Discriminator/{stem_name}/Segment', loss_d.item(), global_step)
                    writer.add_scalar(f'Metrics/SDR/{stem_name}/Segment', sdr, global_step)
                    writer.add_scalar(f'Metrics/SIR/{stem_name}/Segment', sir, global_step)
                    writer.add_scalar(f'Metrics/SAR/{stem_name}/Segment', sar, global_step)

                # Print segment-based metrics
                print(f"Segment [{segment_count}/{total_segments}] - Loss G: {float(loss_g):.4f}, Loss D: {float(loss_d):.4f}, SDR: {float(sdr):.4f}, SIR: {float(sir):.4f}, SAR: {float(sar):.4f}")

                full_outputs.append(outputs)

                # Checkpoint saving after each segment if flag is set
                if checkpoint_flag.value == 1:
                    logger.info(f"Segment checkpoint triggered at segment {segment_count}.")
                    save_checkpoint(
                        model=model,
                        discriminator=discriminator,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        scaler_g=scaler_g,
                        scaler_d=scaler_d,
                        epoch=epoch,
                        loss_g=running_loss_g,
                        loss_d=running_loss_d,
                        checkpoint_dir=training_params['checkpoint_dir'],
                        stem_name=stem_name,
                        segment=segment_count,  # Pass the current segment count
                        data_dir=training_params['data_dir'],  # Ensure data_dir is passed
                        batch_size=training_params['batch_size'],
                        num_epochs=training_params['num_epochs'],
                        save_interval=training_params['save_interval'],
                        accumulation_steps=training_params['accumulation_steps'],
                        num_stems=training_params['num_stems'],
                        num_workers=training_params['num_workers'],
                        cache_dir=training_params['cache_dir'],
                        segments_per_track=training_params['segments_per_track'],
                        loss_function_g=model_params['loss_function_g'],
                        loss_function_d=model_params['loss_function_d'],
                        perceptual_loss_flag=model_params['perceptual_loss_flag'],
                        perceptual_loss_weight=model_params['perceptual_loss_weight'],
                        clip_value=model_params['clip_value'],
                        scheduler_step_size=training_params['scheduler_step_size'],
                        scheduler_gamma=training_params['scheduler_gamma'],
                        tensorboard_flag=model_params['tensorboard_flag'],
                        add_noise=training_params['add_noise'],
                        noise_amount=training_params['noise_amount'],
                        early_stopping_patience=training_params['early_stopping_patience'],
                        disable_early_stopping=training_params['disable_early_stopping'],
                        suppress_warnings=training_params['suppress_warnings'],
                        suppress_reading_messages=training_params['suppress_reading_messages'],
                        discriminator_update_interval=training_params['discriminator_update_interval'],
                        label_smoothing_real=training_params['label_smoothing_real'],
                        label_smoothing_fake=training_params['label_smoothing_fake'],
                        suppress_detailed_logs=training_params['suppress_detailed_logs'],
                        use_cache=training_params['use_cache'],
                        channel_multiplier=model_params['channel_multiplier'],
                        update_cache=training_params['update_cache']
                    )
                    checkpoint_flag.value = 0  # Reset the flag after saving

            # Reset the segment count to 0 after each file
            segment_count = 0

            # Assemble the full output for the entire track
            full_output = torch.cat(full_outputs, dim=0)

            # Perform necessary processing with full_output, such as saving or further evaluation
            # Example: Save the output to a file
            # save_output(full_output, file_id['identifier'], stem_name)

            logger.info(f"Completed processing file_id: {file_id['identifier']} for stem: {stem_name}")

        # Log running losses
        avg_loss_g = running_loss_g / len(dataset)
        avg_loss_d = running_loss_d / len(dataset)

        logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}] - Average Generator Loss: {avg_loss_g:.4f}, Average Discriminator Loss: {avg_loss_d:.4f}")

        if model_params['tensorboard_flag']:
            writer.add_scalar(f'Loss/Generator/{stem_name}/Epoch', avg_loss_g, epoch)
            writer.add_scalar(f'Loss/Discriminator/{stem_name}/Epoch', avg_loss_d, epoch)

        # Validate the model
        model.eval()
        val_loss_g = 0.0
        val_loss_d = 0.0

        with torch.no_grad():
            for i, file_id in enumerate(val_dataset.file_ids):
                input_file_path = os.path.join(val_dataset.data_dir, file_id['input_file'])
                target_file_path = os.path.join(val_dataset.data_dir, file_id['target_files'][stem_name])

                input_data = val_dataset.process_and_cache_file(input_file_path, file_id['identifier'], 'input')
                target_data = val_dataset.process_and_cache_file(target_file_path, file_id['identifier'], stem_name)

                if input_data.numel() == 0 or target_data.numel() == 0:
                    logger.warning(f"Skipping empty input or target for file_id: {file_id['identifier']}")
                    continue

                for segment_idx in range(input_data.size(0)):
                    input_segment = input_data[segment_idx].unsqueeze(0)
                    target_segment = target_data[segment_idx].unsqueeze(0)

                    if model_params['tensorboard_flag']:
                        logger.debug(f"Input tensor shape: {input_segment.shape}")
                        logger.debug(f"Target tensor shape: {target_segment.shape}")

                    with autocast():
                        outputs = model(input_segment.to(device))
                        if model_params['tensorboard_flag']:
                            logger.debug(f"Model output tensor shape: {outputs.shape}")

                        # Ensure target tensor shape matches output tensor shape
                        if target_segment.dim() == 5 and target_segment.size(1) == 1:
                            target_segment = target_segment.squeeze(1)

                        validate_tensor_shapes(outputs, target_segment, "Before computing val_loss_g")
                        val_loss_g += model_params['loss_function_g'](outputs, target_segment).item()

                        real_out = discriminator(target_segment)
                        fake_out = discriminator(outputs)
                        loss_d_real = model_params['loss_function_d'](real_out, torch.ones_like(real_out) * training_params['label_smoothing_real'])
                        loss_d_fake = model_params['loss_function_d'](fake_out, torch.zeros_like(fake_out) * training_params['label_smoothing_fake'])
                        val_loss_d += (loss_d_real.item() + loss_d_fake.item()) / 2

            avg_val_loss_g = val_loss_g / len(val_dataset)
            avg_val_loss_d = val_loss_d / len(val_dataset)

            if model_params['tensorboard_flag']:
                writer.add_scalar(f'Loss/Generator/{stem_name}/Validation', avg_val_loss_g, epoch)
                writer.add_scalar(f'Loss/Discriminator/{stem_name}/Validation', avg_val_loss_d, epoch)

            logger.info(f"Validation - Average Generator Loss: {avg_val_loss_g:.4f}, Average Discriminator Loss: {avg_val_loss_d:.4f}")

            scheduler_g.step(avg_val_loss_g)
            scheduler_d.step(avg_val_loss_d)

            if best_val_loss is None or avg_val_loss_g < best_val_loss:
                best_val_loss = avg_val_loss_g
                early_stopping_counter = 0
                save_checkpoint(
                    model=model,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    epoch=epoch,
                    loss_g=avg_val_loss_g,
                    loss_d=avg_val_loss_d,
                    checkpoint_dir=training_params['checkpoint_dir'],
                    stem_name=stem_name,
                    is_epoch_checkpoint=True,
                    data_dir=training_params['data_dir'],  # Ensure data_dir is passed
                    batch_size=training_params['batch_size'],
                    num_epochs=training_params['num_epochs'],
                    save_interval=training_params['save_interval'],
                    accumulation_steps=training_params['accumulation_steps'],
                    num_stems=training_params['num_stems'],
                    num_workers=training_params['num_workers'],
                    cache_dir=training_params['cache_dir'],
                    segments_per_track=training_params['segments_per_track'],
                    loss_function_g=model_params['loss_function_g'],
                    loss_function_d=model_params['loss_function_d'],
                    perceptual_loss_flag=model_params['perceptual_loss_flag'],
                    perceptual_loss_weight=model_params['perceptual_loss_weight'],
                    clip_value=model_params['clip_value'],
                    scheduler_step_size=training_params['scheduler_step_size'],
                    scheduler_gamma=training_params['scheduler_gamma'],
                    tensorboard_flag=model_params['tensorboard_flag'],
                    add_noise=training_params['add_noise'],
                    noise_amount=training_params['noise_amount'],
                    early_stopping_patience=training_params['early_stopping_patience'],
                    disable_early_stopping=training_params['disable_early_stopping'],
                    suppress_warnings=training_params['suppress_warnings'],
                    suppress_reading_messages=training_params['suppress_reading_messages'],
                    discriminator_update_interval=training_params['discriminator_update_interval'],
                    label_smoothing_real=training_params['label_smoothing_real'],
                    label_smoothing_fake=training_params['label_smoothing_fake'],
                    suppress_detailed_logs=training_params['suppress_detailed_logs'],
                    use_cache=training_params['use_cache'],
                    channel_multiplier=model_params['channel_multiplier'],
                    update_cache=training_params['update_cache']
                )
                logger.info(f"Saved best model checkpoint for epoch: {epoch + 1}")

            else:
                early_stopping_counter += 1
                if early_stopping_counter >= training_params['early_stopping_patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    return

        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

    if writer:
        writer.close()

    logger.info(f"Completed training for single stem: {stem_name}")

def save_checkpoint(model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d, epoch, loss_g, loss_d, checkpoint_dir, stem_name, segment=None, **kwargs):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scaler_g_state_dict': scaler_g.state_dict(),
        'scaler_d_state_dict': scaler_d.state_dict(),
        'epoch': epoch,
        'loss_g': loss_g,
        'loss_d': loss_d,
        'stem_name': stem_name,
        'n_mels': model.n_mels,
        'target_length': model.target_length,  # Save target_length
        'optimizer_name_g': optimizer_g.__class__.__name__,
        'optimizer_name_d': optimizer_d.__class__.__name__,
        'initial_lr_g': optimizer_g.param_groups[0]['initial_lr'],
        'initial_lr_d': optimizer_d.param_groups[0]['initial_lr'],
        'weight_decay': optimizer_g.param_groups[0]['weight_decay'],
        'batch_size': kwargs.get('batch_size'),
        'num_epochs': kwargs.get('num_epochs'),
        'save_interval': kwargs.get('save_interval'),
        'accumulation_steps': kwargs.get('accumulation_steps'),
        'num_stems': kwargs.get('num_stems'),
        'num_workers': kwargs.get('num_workers'),
        'cache_dir': kwargs.get('cache_dir'),
        'segments_per_track': kwargs.get('segments_per_track'),
        'loss_function_g': kwargs.get('loss_function_g').__class__.__name__,
        'loss_function_d': kwargs.get('loss_function_d').__class__.__name__,
        'perceptual_loss_flag': kwargs.get('perceptual_loss_flag'),
        'perceptual_loss_weight': kwargs.get('perceptual_loss_weight'),
        'clip_value': kwargs.get('clip_value'),
        'scheduler_step_size': kwargs.get('scheduler_step_size'),
        'scheduler_gamma': kwargs.get('scheduler_gamma'),
        'tensorboard_flag': kwargs.get('tensorboard_flag'),
        'add_noise': kwargs.get('add_noise'),
        'noise_amount': kwargs.get('noise_amount'),
        'early_stopping_patience': kwargs.get('early_stopping_patience'),
        'disable_early_stopping': kwargs.get('disable_early_stopping'),
        'suppress_warnings': kwargs.get('suppress_warnings'),
        'suppress_reading_messages': kwargs.get('suppress_reading_messages'),
        'channel_multiplier': kwargs.get('channel_multiplier'),
        'update_cache': kwargs.get('update_cache'),
        'segment': segment
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_segment_{segment}.pt' if segment is not None else f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def save_checkpoint_gradio(checkpoint_dir, training_state, checkpoint_flag):
    if not training_state['training_started']:
        return "Training has not started yet. Cannot save checkpoint."

    checkpoint_flag.value = 1  # Set the flag to trigger checkpoint saving

    # Wait for the checkpoint to be saved (with a timeout)
    timeout = 60  # seconds
    start_time = time.time()
    while checkpoint_flag.value == 1:
        time.sleep(0.1)
        if time.time() - start_time > timeout:
            return "Checkpoint saving timed out. Checkpoint may be saved anyway, Check. The training process might be busy or unresponsive."

    return f"Checkpoint saving triggered. It will be saved after the current segment completes."

def start_training(
    data_dir: str, batch_size: int, num_epochs: int, initial_lr_g: float, initial_lr_d: float,
    use_cuda: bool, checkpoint_dir: str, save_interval: int, accumulation_steps: int, num_stems: int,
    num_workers: int, cache_dir: str, loss_function_g: nn.Module, loss_function_d: nn.Module,
    optimizer_name_g: str, optimizer_name_d: str, perceptual_loss_flag: bool, perceptual_loss_weight: float,
    clip_value: float, scheduler_step_size: int, scheduler_gamma: float, tensorboard_flag: bool,
    add_noise: bool, noise_amount: float, early_stopping_patience: int,
    disable_early_stopping: bool, weight_decay: float, suppress_warnings: bool, suppress_reading_messages: bool,
    discriminator_update_interval: int, label_smoothing_real: float, label_smoothing_fake: float,
    suppress_detailed_logs: bool, stop_flag: Value, checkpoint_flag: Value, training_state: dict,
    channel_multiplier: float, segments_per_track: int = 5, use_cache: bool = True, update_cache: bool = False,
    current_segment: int = 0
):
    # Log the training parameters to confirm they are being used correctly
    logger.info(f"Training Parameters from training_state:")
    logger.info(training_state)

    # Override function parameters with those from training_state
    training_params = {
        'data_dir': data_dir,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'initial_lr_g': initial_lr_g,
        'initial_lr_d': initial_lr_d,
        'use_cuda': use_cuda,
        'checkpoint_dir': checkpoint_dir,
        'save_interval': save_interval,
        'accumulation_steps': accumulation_steps,
        'num_stems': num_stems,
        'num_workers': num_workers,
        'cache_dir': cache_dir,
        'segments_per_track': segments_per_track,
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
        'use_cache': use_cache,
        'update_cache': update_cache,
        'device_str': 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    }

    model_params = {
        'optimizer_name_g': optimizer_name_g.capitalize(),
        'optimizer_name_d': optimizer_name_d.capitalize(),
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
        device=training_params['device_str'],
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
        device=training_params['device_str'],
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
        device=training_params['device_str'],
        segments_per_track=segments_per_track,
        file_ids=val_file_ids
    )

    # Ensure training_state is a dictionary-like object
    training_state = dict(training_state) if isinstance(training_state, DictProxy) else training_state

    for stem_name in ['vocals', 'drums', 'bass', 'kick', 'keys', 'guitar']:
        if stop_flag.value == 1:
            logger.info("Training stopped.")
            return

        if stem_name == "input":
            continue

        start_time = time.time()

        # Debugging statement
        print(f"[start_training] training_state type: {type(training_state)}")
        print(f"[start_training] training_state value: {training_state}")

        train_single_stem(
            stem_name, train_dataset, val_dataset, training_params, model_params,
            sample_rate, n_mels, n_fft, segment_length, stop_flag, checkpoint_flag, training_state, suppress_reading_messages,  # Pass training_state
            current_segment  # Pass current_segment
        )

        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Training for stem {stem_name} finished in {epoch_duration:.2f} seconds")

    logger.info("Training finished.")

def start_training_wrapper(
    data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
    accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g,
    optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma,
    tensorboard_flag, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay,
    suppress_warnings, suppress_reading_messages, discriminator_update_interval, label_smoothing_real, label_smoothing_fake,
    suppress_detailed_logs, use_cache, channel_multiplier, segments_per_track, update_cache, training_state, stop_flag, checkpoint_flag
):
    global training_process
    stop_flag.value = 0  # Reset stop flag
    checkpoint_flag.value = 0  # Reset checkpoint flag

    # Ensure proper capitalization for optimizers
    optimizer_name_g = optimizer_name_g.capitalize()
    optimizer_name_d = optimizer_name_d.capitalize()

    print(f"[start_training_wrapper] training_state type before assert: {type(training_state)}")
    print(f"[start_training_wrapper] training_state value before assert: {training_state}")

    assert isinstance(training_state, (dict, DictProxy)), "training_state must be a dictionary or DictProxy object"

    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "WassersteinLoss": wasserstein_loss
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]

    # Update training_state with parameters
    training_state.update({
        'data_dir': data_dir,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'initial_lr_g': learning_rate_g,
        'initial_lr_d': learning_rate_d,
        'use_cuda': use_cuda,
        'checkpoint_dir': checkpoint_dir,
        'save_interval': save_interval,
        'accumulation_steps': accumulation_steps,
        'num_stems': num_stems,
        'num_workers': num_workers,
        'cache_dir': cache_dir,
        'segments_per_track': segments_per_track,
        'loss_function_g': loss_function_g,
        'loss_function_d': loss_function_d,
        'optimizer_name_g': optimizer_name_g,
        'optimizer_name_d': optimizer_name_d,
        'perceptual_loss_flag': perceptual_loss_flag,
        'clip_value': clip_value,
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
        'perceptual_loss_weight': perceptual_loss_weight,
        'suppress_detailed_logs': suppress_detailed_logs,
        'use_cache': use_cache,
        'channel_multiplier': channel_multiplier,
        'update_cache': update_cache
    })

    training_process = Process(target=start_training, args=(
        data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
        accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
        perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise,
        noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
        discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, stop_flag, checkpoint_flag,
        training_state, use_cache, channel_multiplier, segments_per_track, update_cache))

    print(f"[start_training_wrapper] training_state type before process start: {type(training_state)}")
    print(f"[start_training_wrapper] training_state value before process start: {training_state}")

    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"

def stop_training_wrapper(stop_flag):
    global training_process
    if training_process is not None:
        stop_flag.value = 1  # Set stop flag to request stopping
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def resume_training(checkpoint_dir, device_str, stop_flag, checkpoint_flag, training_state):
    logger.info(f"Resuming training from checkpoint at {checkpoint_dir}")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        return "No checkpoints found in the specified directory."

    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    try:
        checkpoint = load_checkpoint(checkpoint_path)
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
        return f"Error loading checkpoint: {e}"

    n_mels = checkpoint.get('n_mels', 32)
    target_length = checkpoint.get('target_length', 22050)
    optimizer_name_g = checkpoint.get('optimizer_name_g', 'Adam')
    optimizer_name_d = checkpoint.get('optimizer_name_d', 'RMSprop')
    initial_lr_g = checkpoint.get('initial_lr_g', 1e-4)
    initial_lr_d = checkpoint.get('initial_lr_d', 1e-4)
    weight_decay = checkpoint.get('weight_decay', 1e-5)

    model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = create_model_and_optimizer(
        device=device_str,
        n_mels=n_mels,
        target_length=target_length,
        initial_lr_g=initial_lr_g,
        initial_lr_d=initial_lr_d,
        optimizer_name_g=optimizer_name_g,
        optimizer_name_d=optimizer_name_d,
        weight_decay=weight_decay
    )

    if 'model_state_dict' in checkpoint:
        filtered_checkpoint = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model.state_dict()}
        model.load_state_dict(filtered_checkpoint, strict=False)
    else:
        logger.error("Checkpoint does not contain model_state_dict")
        return "Checkpoint does not contain model_state_dict"

    if 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
    else:
        logger.error("Checkpoint does not contain discriminator_state_dict")
        return "Checkpoint does not contain discriminator_state_dict"

    if 'optimizer_g_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    else:
        logger.error("Checkpoint does not contain optimizer_g_state_dict")
        return "Checkpoint does not contain optimizer_g_state_dict"

    if 'optimizer_d_state_dict' in checkpoint:
        try:
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        except ValueError as e:
            logger.warning(f"Error loading optimizer_d state_dict: {e}. Attempting to adapt state_dict.")
            optimizer_d_state = checkpoint['optimizer_d_state_dict']
            optimizer_d_state['param_groups'] = optimizer_d.state_dict()['param_groups']
            optimizer_d.load_state_dict(optimizer_d_state)
    else:
        logger.error("Checkpoint does not contain optimizer_d_state_dict")
        return "Checkpoint does not contain optimizer_d_state_dict"

    if 'scaler_g_state_dict' in checkpoint:
        scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
    else:
        logger.error("Checkpoint does not contain scaler_g_state_dict")
        return "Checkpoint does not contain scaler_g_state_dict"

    if 'scaler_d_state_dict' in checkpoint:
        scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
    else:
        logger.error("Checkpoint does not contain scaler_d_state_dict")
        return "Checkpoint does not contain scaler_d_state_dict"

    epoch = checkpoint['epoch']
    segment = checkpoint.get('segment', 0)

    start_training(
        data_dir=checkpoint['data_dir'],
        batch_size=checkpoint['batch_size'],
        num_epochs=checkpoint['num_epochs'] - epoch,
        initial_lr_g=initial_lr_g,
        initial_lr_d=initial_lr_d,
        use_cuda='cuda' in device_str,
        checkpoint_dir=checkpoint['checkpoint_dir'],
        save_interval=checkpoint['save_interval'],
        accumulation_steps=checkpoint['accumulation_steps'],
        num_stems=checkpoint['num_stems'],
        num_workers=checkpoint['num_workers'],
        cache_dir=checkpoint['cache_dir'],
        loss_function_g=checkpoint['loss_function_g'],
        loss_function_d=checkpoint['loss_function_d'],
        optimizer_name_g=optimizer_name_g,
        optimizer_name_d=optimizer_name_d,
        perceptual_loss_flag=checkpoint['perceptual_loss_flag'],
        perceptual_loss_weight=checkpoint['perceptual_loss_weight'],
        clip_value=checkpoint['clip_value'],
        scheduler_step_size=checkpoint['scheduler_step_size'],
        scheduler_gamma=checkpoint['scheduler_gamma'],
        tensorboard_flag=checkpoint['tensorboard_flag'],
        add_noise=checkpoint['add_noise'],
        noise_amount=checkpoint['noise_amount'],
        early_stopping_patience=checkpoint['early_stopping_patience'],
        disable_early_stopping=checkpoint['disable_early_stopping'],
        weight_decay=weight_decay,
        suppress_warnings=checkpoint['suppress_warnings'],
        suppress_reading_messages=checkpoint['suppress_reading_messages'],
        channel_multiplier=checkpoint['channel_multiplier'],
        segments_per_track=checkpoint.get('segments_per_track', 10),
        stop_flag=stop_flag,
        update_cache=checkpoint.get('update_cache', True),
        training_state=training_state
    )

    return f"Resumed training from checkpoint: {latest_checkpoint}"
