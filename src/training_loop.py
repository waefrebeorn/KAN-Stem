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
from data_preprocessing import preprocess_and_cache_dataset
from model_setup import create_model_and_optimizer
from dataset import StemSeparationDataset, collate_fn
from utils import compute_sdr, compute_sir, compute_sar, log_training_parameters, detect_parameters

logger = logging.getLogger(__name__)

def train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, device_str, checkpoint_dir, save_interval, accumulation_steps, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, loss_function_g, loss_function_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, num_workers, early_stopping_patience, disable_early_stopping, weight_decay, use_cpu_for_prep, suppress_warnings, suppress_reading_messages, cache_dir, device_prep, add_noise, noise_amount, discriminator_update_interval, label_smoothing_real, label_smoothing_fake):
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(device_str, n_mels, target_length, cache_dir, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, weight_decay)

    feature_extractor = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device_str).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    train_loader = DataLoader(
        StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_warnings=suppress_warnings, num_workers=num_workers, device_prep=device_prep),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Updated to use ReduceLROnPlateau scheduler
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=scheduler_gamma, patience=scheduler_step_size)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=scheduler_gamma, patience=scheduler_step_size)

    scaler = GradScaler()
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

            inputs = data['input'].to(device_str, non_blocking=True)
            targets = data['target'][:, stem].to(device_str, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                # Ensure outputs and targets have the same number of channels
                outputs = outputs[:, :targets.size(1), :, :]
                
                targets = targets.unsqueeze(1)
                outputs = outputs.unsqueeze(1)

                if outputs.dim() == 5:
                    outputs = outputs.squeeze(2)
                if targets.dim() == 5:
                    targets = targets.squeeze(2)

                loss_g = loss_function_g(outputs.to(device_str), targets.to(device_str))
                
                if perceptual_loss_flag:
                    outputs_3ch = convert_to_3_channels(outputs)
                    targets_3ch = convert_to_3_channels(targets)
                    perceptual_loss = PerceptualLoss(feature_extractor)(outputs_3ch.to(device_str), targets_3ch.to(device_str)) * perceptual_loss_weight
                    loss_g += perceptual_loss
                
                scaler.scale(loss_g).backward(retain_graph=True)

                if (i + 1) % discriminator_update_interval == 0:
                    # Add noise to the discriminator's input
                    if add_noise:
                        noise = torch.randn_like(targets) * noise_amount
                        targets = targets + noise

                    # Apply label smoothing
                    real_labels = torch.full((inputs.size(0), 1), label_smoothing_real, device=device_str, dtype=torch.float)
                    fake_labels = torch.full((inputs.size(0), 1), label_smoothing_fake, device=device_str, dtype=torch.float)
                    real_out = discriminator(targets.to(device_str).clone().detach())
                    fake_out = discriminator(outputs.clone().detach())

                    loss_d_real = loss_function_d(real_out, real_labels)
                    loss_d_fake = loss_function_d(fake_out, fake_labels)
                    gp = gradient_penalty(discriminator, targets.to(device_str), outputs.to(device_str), device_str)
                    loss_d = (loss_d_real + loss_d_fake) / 2 + gp  # Adding gradient penalty term

                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_d.zero_grad()
                    running_loss_d += loss_d.item()

                running_loss_g += loss_g.item()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    scaler.step(optimizer_g)
                    scaler.update()
                    optimizer_g.zero_grad()

                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g / (i + 1):.4f}, Loss D: {running_loss_d / (i + 1):.4f}')
                    if tensorboard_flag:
                        writer.add_scalar('Loss/Generator', running_loss_g / (i + 1), epoch * len(train_loader) + i)
                        writer.add_scalar('Loss/Discriminator', running_loss_d / (i + 1), epoch * len(train_loader) + i)

        optimizer_g.step()
        optimizer_d.step()

        # Validation step only after training epoch completes
        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        num_sdr_samples, num_sir_samples, num_sar_samples = 0, 0, 0

        with torch.no_grad():
            try:
                for i, data in enumerate(val_loader):
                    if data is None:
                        continue

                    logger.debug(f"Validating batch {i+1}/{len(val_loader)}")

                    inputs = data['input'].to(device_str, non_blocking=True)
                    targets = data['target'][:, stem].to(device_str, non_blocking=True)
                    outputs = model(inputs)
                    target_length = targets.size(-1)
                    outputs = outputs[..., :target_length]

                    # Ensure outputs and targets have the same number of channels
                    outputs = outputs[:, :targets.size(1), :, :]
                    
                    targets = targets.unsqueeze(1)
                    outputs = outputs.unsqueeze(1)

                    if outputs.dim() == 5:
                        outputs = outputs.squeeze(2)
                    if targets.dim() == 5:
                        targets = targets.squeeze(2)

                    loss = loss_function_g(outputs, targets)
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
                sdr_avg = sdr_total / max(num_sdr_samples, 1)  # Avoid division by zero
                sir_avg = sir_total / max(num_sir_samples, 1)  # Avoid division by zero
                sar_avg = sar_total / max(num_sar_samples, 1)  # Avoid division by zero

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

                if not disable_early_stopping and early_stopping_counter >= early_stopping_patience:
                    logger.info('Early stopping triggered.')
                    break
            except Exception as e:
                logger.error(f"Error during validation step: {e}", exc_info=True)

        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

    final_model_path = f"{checkpoint_dir}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if tensorboard_flag:
        writer.close()
