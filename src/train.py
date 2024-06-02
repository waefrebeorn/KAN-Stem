import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import torch.multiprocessing as mp
import logging

from dataset import StemSeparationDataset, collate_fn
from utils import detect_parameters, get_optimizer, compute_adversarial_loss, PerceptualLoss
from model import KANWithDepthwiseConv, KANDiscriminator

mp.set_start_method('spawn', force=True)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature under heavy development")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s')

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

def start_training(data_dir, val_dir, batch_size, num_epochs, initial_lr_g, initial_lr_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    global device
    logger.info(f"Starting training with dataset at {data_dir}")
    device_str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    device_prep = 'cpu' if use_cpu_for_prep else device_str
    logger.info(f"Using device: {device_str} for training and {device_prep} for preprocessing")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    preprocess_and_cache_dataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device_prep)
    
    for stem in range(num_stems):
        try:
            train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, device_str, checkpoint_dir, save_interval, accumulation_steps, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, loss_function_g, loss_function_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, num_workers, early_stopping_patience, weight_decay, use_cpu_for_prep, suppress_warnings, suppress_reading_messages, cache_dir)
        except Exception as e:
            logger.error(f"Error in train_single_stem: {e}", exc_info=True)
            raise  # Reraise the exception to halt training on error

def train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, device_str, checkpoint_dir, save_interval, accumulation_steps, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, loss_function_g, loss_function_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, num_workers, early_stopping_patience, weight_decay, use_cpu_for_prep, suppress_warnings, suppress_reading_messages, cache_dir):
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(device_str, n_mels, target_length, cache_dir, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, weight_decay)

    train_loader = DataLoader(
        StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, use_cpu_for_prep),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_warnings=suppress_warnings, num_workers=num_workers, device_prep=use_cpu_for_prep),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

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

            inputs = data['input'].to(device_str)
            targets = data['target'][:, stem].to(device_str)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                targets = targets.unsqueeze(1)
                outputs = outputs.unsqueeze(1)

                if outputs.dim() == 5:
                    outputs = outputs.squeeze(2)
                if targets.dim() == 5:
                    targets = targets.squeeze(2)

                loss_g = loss_function_g(outputs.to(device_str), targets.to(device_str))
                scaler.scale(loss_g).backward(retain_graph=True)

                real_labels = torch.ones(inputs.size(0), 1, device=device_str)
                fake_labels = torch.zeros(inputs.size(0), 1, device=device_str)
                real_out = discriminator(targets.to(device_str).clone().detach())
                fake_out = discriminator(outputs.clone().detach())

                loss_d_real = loss_function_d(real_out, real_labels)
                loss_d_fake = loss_function_d(fake_out, fake_labels)
                gp = gradient_penalty(discriminator, targets.to(device_str), outputs.to(device_str), device_str)
                loss_d = (loss_d_real + loss_d_fake) / 2 + 10 * gp  # Adding gradient penalty term

                scaler.scale(loss_d).backward()

                running_loss_g += loss_g.item()
                running_loss_d += loss_d.item()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                    scaler.step(optimizer_g)
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()

                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g / (i + 1):.4f}, Loss D: {running_loss_d / (i + 1):.4f}')
                    if tensorboard_flag:
                        writer.add_scalar('Loss/Generator', running_loss_g / (i + 1), epoch * len(train_loader) + i)
                        writer.add_scalar('Loss/Discriminator', running_loss_d / (i + 1), epoch * len(train_loader) + i)

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        # Validation step only after training epoch completes
        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            try:
                for i, data in enumerate(val_loader):
                    if data is None:
                        continue

                    logger.debug(f"Validating batch {i+1}/{len(val_loader)}")

                    inputs = data['input'].to(device_str)
                    targets = data['target'][:, stem].to(device_str)
                    outputs = model(inputs)
                    target_length = targets.size(-1)
                    outputs = outputs[..., :target_length]

                    targets = targets.unsqueeze(1)
                    outputs = outputs.unsqueeze(1)

                    if outputs.dim() == 5:
                        outputs = outputs.squeeze(2)
                    if targets.dim() == 5:
                        targets = targets.squeeze(2)

                    loss = loss_function_g(outputs, targets)
                    val_loss += loss.item()

                    for j in range(outputs.size(0)):
                        real_out = discriminator(targets[j].unsqueeze(0))
                        fake_out = discriminator(outputs[j].unsqueeze(0))

                        real_labels = torch.ones(real_out.size(0), 1, device=device_str)
                        fake_labels = torch.zeros(fake_out.size(0), 1, device=device_str)

                        real_loss = F.binary_cross_entropy_with_logits(real_out, real_labels)
                        fake_loss = F.binary_cross_entropy_with_logits(fake_out, fake_labels)
                        sdr = (real_loss + fake_loss) / 2

                        if not torch.isnan(sdr):
                            sdr_total += sdr.mean()
                            sir_total += 0.0  # Placeholder, update with actual computation if needed
                            sar_total += 0.0  # Placeholder, update with actual computation if needed

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
            except Exception as e:
                logger.error(f"Error during validation step: {e}", exc_info=True)

    final_model_path = f"{checkpoint_dir}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if tensorboard_flag:
        writer.close()

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep, suppress_detailed_logs):
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
    training_process = mp.Process(target=start_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"
    
def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

if __name__ == '__main__':
    # Call your start_training_wrapper or any other function here as needed
    pass
