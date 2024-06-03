import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import torch.multiprocessing as mp
import logging
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

from dataset import StemSeparationDataset, collate_fn
from utils import detect_parameters, get_optimizer, PerceptualLoss
from model import KANWithDepthwiseConv, KANDiscriminator

mp.set_start_method('spawn', force=True)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature under heavy development")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_process = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)

    interpolated_out = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=interpolated_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_out),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep):
    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep)
    
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
        except Exception as e:
            logger.error(f"Error during preprocessing and caching: {e}")
    
    logger.info("Preprocessing and caching completed.")

def create_model_and_optimizer(device, n_mels, target_length, cache_dir, learning_rate_g, learning_rate_d, optimizer_name_g, optimizer_name_d, weight_decay):
    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d, weight_decay)
    return model, discriminator, optimizer_g, optimizer_d

def log_training_parameters(params):
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

def convert_to_3_channels(tensor):
    return tensor.expand(-1, 3, -1, -1)

def start_training(data_dir, val_dir, batch_size, num_epochs, initial_lr_g, initial_lr_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs):
    global device

    logger.info(f"Starting training with dataset at {data_dir}")
    device_str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    device_prep = 'cpu' if use_cpu_for_prep else device_str
    logger.info(f"Using device: {device_str} for training and {device_prep} for preprocessing")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep)
    
    # Log the parameters after preprocessing
    params = {
        "Data Directory": data_dir,
        "Validation Directory": val_dir,
        "Batch Size": batch_size,
        "Number of Epochs": num_epochs,
        "Generator Learning Rate": initial_lr_g,
        "Discriminator Learning Rate": initial_lr_d,
        "Use CUDA": use_cuda,
        "Checkpoint Directory": checkpoint_dir,
        "Save Interval": save_interval,
        "Accumulation Steps": accumulation_steps,
        "Number of Stems": num_stems,
        "Number of Workers": num_workers,
        "Cache Directory": cache_dir,
        "Generator Loss Function": str(loss_function_g),
        "Discriminator Loss Function": str(loss_function_d),
        "Generator Optimizer": optimizer_name_g,
        "Discriminator Optimizer": optimizer_name_d,
        "Use Perceptual Loss": perceptual_loss_flag,
        "Perceptual Loss Weight": perceptual_loss_weight,
        "Gradient Clipping Value": clip_value,
        "Scheduler Step Size": scheduler_step_size,
        "Scheduler Gamma": scheduler_gamma,
        "Enable TensorBoard Logging": tensorboard_flag,
        "Apply Data Augmentation": apply_data_augmentation,
        "Add Noise": add_noise,
        "Noise Amount": noise_amount,
        "Early Stopping Patience": early_stopping_patience,
        "Disable Early Stopping": disable_early_stopping,
        "Weight Decay": weight_decay,
        "Suppress Warnings": suppress_warnings,
        "Suppress Reading Messages": suppress_reading_messages,
        "Use CPU for Preparation": use_cpu_for_prep,
        "Discriminator Update Interval": discriminator_update_interval,
        "Label Smoothing Real": label_smoothing_real,
        "Label Smoothing Fake": label_smoothing_fake,
        "Suppress Detailed Logs": suppress_detailed_logs
    }
    log_training_parameters(params)

    for stem in range(num_stems):
        try:
            train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, device_str, checkpoint_dir, save_interval, accumulation_steps, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, loss_function_g, loss_function_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, num_workers, early_stopping_patience, disable_early_stopping, weight_decay, use_cpu_for_prep, suppress_warnings, suppress_reading_messages, cache_dir, device_prep, add_noise, noise_amount, discriminator_update_interval, label_smoothing_real, label_smoothing_fake)
        except Exception as e:
            logger.error(f"Error in train_single_stem: {e}", exc_info=True)
            raise  # Reraise the exception to halt training on error

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
                    real_labels = torch.full((inputs.size(0), 1), label_smoothing_real, device=device_str)
                    fake_labels = torch.full((inputs.size(0), 1), label_smoothing_fake, device=device_str)
                    real_out = discriminator(targets.to(device_str).clone().detach())
                    fake_out = discriminator(outputs.clone().detach())

                    loss_d_real = loss_function_d(real_out, real_labels)
                    loss_d_fake = loss_function_d(fake_out, fake_labels)
                    gp = gradient_penalty(discriminator, targets.to(device_str), outputs.to(device_str), device_str)
                    loss_d = (loss_d_real + loss_d_fake) / 2 + gp  # Adding gradient penalty term

                    scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                    scaler.step(optimizer_d)
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

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs):
    global training_process
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "WassersteinLoss": wasserstein_loss
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]
    training_process = mp.Process(target=start_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"
    
def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def compute_sdr(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    sdr = 10 * torch.log10(s_true / (s_noise + 1e-8))
    return sdr

def compute_sir(true, pred):
    noise = true - pred
    s_true = torch.sum(true ** 2, dim=[1, 2, 3])
    s_interf = torch.sum((true - noise) ** 2, dim=[1, 2, 3])
    sir = 10 * torch.log10(s_true / (s_interf + 1e-8))
    return sir

def compute_sar(true, pred):
    noise = true - pred
    s_noise = torch.sum(noise ** 2, dim=[1, 2, 3])
    s_artif = torch.sum((pred - noise) ** 2, dim=[1, 2, 3])
    sar = 10 * torch.log10(s_noise / (s_artif + 1e-8))
    return sar

if __name__ == '__main__':
    # Call your start_training_wrapper or any other function here as needed
    pass
