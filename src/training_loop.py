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
from torch.optim.lr_scheduler import CyclicLR
import optuna
from data_preprocessing import preprocess_and_cache_dataset
from model_setup import create_model_and_optimizer
from dataset import StemSeparationDataset, collate_fn
from utils import compute_sdr, compute_sir, compute_sar, log_training_parameters, detect_parameters, convert_to_3_channels, gradient_penalty, PerceptualLoss

import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

logger = logging.getLogger(__name__)

def train_single_stem(stem, dataset, val_dir, training_params, model_params, sample_rate, n_mels, n_fft, target_length, stop_flag, suppress_reading_messages=False):
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(training_params['checkpoint_dir'], 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if model_params['tensorboard_flag'] else None

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(
        training_params['device_str'], n_mels, target_length, training_params['cache_dir'],
        model_params['initial_lr_g'], model_params['initial_lr_d'], model_params['optimizer_name_g'],
        model_params['optimizer_name_d'], model_params['weight_decay']
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
        num_workers=training_params['num_workers'], device_prep=prep_device, stop_flag=stop_flag
    )
    val_dataset.load_all_stems()  # Ensure the validation set is cached

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=training_params['num_workers'],
        collate_fn=collate_fn
    )

    # Implement CyclicLR scheduler
    scheduler_g = CyclicLR(optimizer_g, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2')
    scheduler_d = CyclicLR(optimizer_d, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2')

    scaler = GradScaler()
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

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for i, data in enumerate(train_loader):
            if stop_flag.value == 1:
                logger.info("Training stopped.")
                return

            if data is None:
                continue

            logger.debug(f"Processing batch {i+1}/{len(train_loader)}")

            inputs = data['input'].to(training_params['device_str'], non_blocking=True)
            targets = data['target'].to(training_params['device_str'], non_blocking=True)

            logger.debug(f"Input shape before reshaping: {inputs.shape}")

            # Ensure the inputs have the correct dimensions
            if inputs.dim() == 2:
                inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)
            elif inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 4 and inputs.size(1) != 1:
                inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)

            logger.debug(f"Input shape after reshaping: {inputs.shape}")

            with autocast():
                outputs = model(inputs)
                outputs = outputs[..., :target_length]

                # Ensure targets and outputs have the correct dimensions
                if targets.dim() == 2:
                    targets = targets.view(targets.size(0), 1, n_mels, target_length)
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                if outputs.dim() == 2:
                    outputs = outputs.view(outputs.size(0), 1, n_mels, target_length)
                if outputs.dim() == 3:
                    outputs = outputs.unsqueeze(1)

                logger.debug(f"Output shape before squeezing: {outputs.shape}")
                logger.debug(f"Target shape before squeezing: {targets.shape}")

                if outputs.dim() == 5:
                    outputs = outputs.squeeze(2)
                if targets.dim() == 5:
                    targets = targets.squeeze(2)

                logger.debug(f"Output shape after squeezing: {outputs.shape}")
                logger.debug(f"Target shape after squeezing: {targets.shape}")

                loss_g = model_params['loss_function_g'](outputs.to(training_params['device_str']), targets.to(training_params['device_str']))

                if model_params['perceptual_loss_flag']:
                    outputs_3ch = convert_to_3_channels(outputs)
                    targets_3ch = convert_to_3_channels(targets)
                    perceptual_loss = PerceptualLoss(feature_extractor)(outputs_3ch.to(training_params['device_str']), targets_3ch.to(training_params['device_str'])) * model_params['perceptual_loss_weight']
                    loss_g += perceptual_loss

                scaler.scale(loss_g).backward(retain_graph=True)

                if (i + 1) % training_params['discriminator_update_interval'] == 0:
                    if training_params['add_noise']:
                        noise = torch.randn_like(targets) * training_params['noise_amount']
                        targets = targets + noise

                    real_labels = torch.full((inputs.size(0), 1), training_params['label_smoothing_real'], device=training_params['device_str'], dtype=torch.float)
                    fake_labels = torch.full((inputs.size(0), 1), training_params['label_smoothing_fake'], device=training_params['device_str'], dtype=torch.float)
                    real_out = discriminator(targets.to(training_params['device_str']).clone().detach())
                    fake_out = discriminator(outputs.clone().detach())

                    loss_d_real = model_params['loss_function_d'](real_out, real_labels)
                    loss_d_fake = model_params['loss_function_d'](fake_out, fake_labels)
                    gp = gradient_penalty(discriminator, targets.to(training_params['device_str']), outputs.to(training_params['device_str']), training_params['device_str'])
                    loss_d = (loss_d_real + loss_d_fake) / 2 + gp

                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), model_params['clip_value'])
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_d.zero_grad()
                    running_loss_d += loss_d.item()

                running_loss_g += loss_g.item()

                if (i + 1) % training_params['accumulation_steps'] == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['clip_value'])
                    scaler.step(optimizer_g)
                    scaler.update()
                    optimizer_g.zero_grad()

                    logger.info(f"Epoch [{epoch+1}/{training_params['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g / (i + 1):.4f}, Loss D: {running_loss_d / (i + 1):.4f}")
                    if model_params['tensorboard_flag']:
                        writer.add_scalar('Loss/Generator', running_loss_g / (i + 1), epoch * len(train_loader) + i)
                        writer.add_scalar('Loss/Discriminator', running_loss_d / (i + 1), epoch * len(train_loader) + i)

                # Step the learning rate scheduler after each batch
                scheduler_g.step()
                scheduler_d.step()

        optimizer_g.step()
        optimizer_d.step()

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

                    inputs = data['input'].to(training_params['device_str'], non_blocking=True)
                    targets = data['target'].to(training_params['device_str'], non_blocking=True)

                    logger.debug(f"Validation input shape before reshaping: {inputs.shape}")

                    # Ensure inputs have the correct dimensions
                    if inputs.dim() == 2:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)
                    elif inputs.dim() == 3:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)
                    elif inputs.dim() == 4 and inputs.size(1) != 1:
                        inputs = inputs.view(inputs.size(0), 1, n_mels, target_length)

                    logger.debug(f"Validation input shape after reshaping: {inputs.shape}")

                    outputs = model(inputs)
                    outputs = outputs[..., :target_length]

                    # Ensure targets and outputs have the correct dimensions
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

        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)

        if (epoch + 1) % training_params['save_interval'] == 0:
            checkpoint_path = os.path.join(training_params['checkpoint_dir'], f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

    final_model_path = f"{training_params['checkpoint_dir']}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if model_params['tensorboard_flag']:
        writer.close()
