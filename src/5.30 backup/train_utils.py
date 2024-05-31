import os
import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from multiprocessing import Queue, Process, Value, Lock, current_process
import logging
import queue
import time
import hashlib

from dataset import StemSeparationDataset, collate_fn
from utils import (
    detect_parameters, get_optimizer, custom_loss, write_audio, check_device,
    compute_adversarial_loss, load_and_preprocess, data_augmentation, read_audio
)
from model import KANWithDepthwiseConv, KANDiscriminator

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress urllib3 debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Suppress TensorFlow oneDNN log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global Variables
training_process = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Initialize device globally
num_processed_stems = Value('i', 0)
total_num_stems = Value('i', 0)
dataset_lock = Lock()

# Define the worker function with detailed logging and the device argument included
def worker(input_queue, output_queue, mel_spectrogram_params, data_dir, target_length, device_str, apply_data_augmentation, valid_stems, lock, num_processed_stems):
    device = torch.device(device_str)  # Ensure the device is properly interpreted
    mel_spectrogram = T.MelSpectrogram(**mel_spectrogram_params).to(device)
    process_name = f"Worker-{current_process().pid}"
    logger.info(f"{process_name}: Started with {len(valid_stems)} stems to process.")

    for stem_name in valid_stems:
        try:
            file_path = os.path.join(data_dir, stem_name)
            logger.info(f"{process_name}: Processing file: {file_path}")

            start_time = time.time()
            # Explicitly pass device argument to load_and_preprocess()
            data = load_and_preprocess(
                file_path, mel_spectrogram, target_length, apply_data_augmentation, device
            )
            end_time = time.time()

            if data is not None:
                logger.info(
                    f"{process_name}: Successfully processed {file_path} in {end_time - start_time:.2f} seconds"
                )
                output_queue.put((stem_name, data))
            else:
                logger.warning(f"{process_name}: Failed to process {file_path}")
                output_queue.put((stem_name, None))

            # Increment the global counter (protected by a lock)
            with lock:
                num_processed_stems.value += 1

        except Exception as e:
            logger.error(f"{process_name}: Error processing {file_path}: {e}")
            output_queue.put((stem_name, None))

    output_queue.close()
    logger.info(f"{process_name}: Finished processing. Output queue closed.")

# Load and preprocess function
def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.debug("Starting load and preprocess")
        input_audio, _ = read_audio(file_path, device='cpu')
        if input_audio is None:
            return None

        logger.debug(f"Device for processing: {device}")

        if apply_data_augmentation:
            input_audio = input_audio.float().to(device)
            logger.debug(f"input_audio device: {input_audio.device}")
            input_audio = data_augmentation(input_audio, device=device)
            logger.debug(f"After data augmentation, input_audio device: {input_audio.device}")
        else:
            input_audio = input_audio.float()

        input_mel = mel_spectrogram(input_audio.to(device)).squeeze(0)[:, :target_length]

        input_mel = input_mel.to('cpu')  # Move back to CPU before returning
        logger.debug(f"input_mel device: {input_mel.device}")

        logger.debug("Completed load and preprocess")
        return input_mel

    except Exception as e:
        logger.error(f"Error in load and preprocess: {e}")
        return None

def create_model_and_optimizer(device, n_mels, target_length, cache_dir, learning_rate_g, learning_rate_d, optimizer_name_g, optimizer_name_d, weight_decay):
    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, cache_dir=cache_dir, device=device).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device).to(device)
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), learning_rate_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), learning_rate_d, weight_decay)
    return model, discriminator, optimizer_g, optimizer_d

def run_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep)

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    global training_process
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]
    training_process = Process(target=run_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"

def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    global device  # Use the global device variable
    logger.info(f"Starting training with dataset at {data_dir}")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256
    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(device, n_mels, target_length, cache_dir, learning_rate_g, learning_rate_d, optimizer_name_g, optimizer_name_d, weight_decay)

    for stem in range(num_stems):
        train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, device, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, model, discriminator, optimizer_g, optimizer_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep)

def train_single_stem(stem, data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, device, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, model, discriminator, optimizer_g, optimizer_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep):
    logger.info("Starting training for single stem: %s", stem)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'runs', f'stem_{stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')) if tensorboard_flag else None

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256

    dataset = StemSeparationDataset(data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation, suppress_warnings, num_workers, device='cpu' if use_cpu_for_prep else device)
    logger.info("Processing stems for training dataset.")
    dataset._process_stems()
    train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    logger.info("Finished processing stems for training dataset.")

    val_dataset = StemSeparationDataset(val_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=suppress_warnings, num_workers=num_workers, device='cpu' if use_cpu_for_prep else device)
    logger.info("Processing stems for validation dataset.")
    val_dataset._process_stems()
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    logger.info("Finished processing stems for validation dataset.")

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

            # --- GPU Memory Management ---
            _, available_memory = torch.cuda.mem_get_info()
            input_size = data['input'].numel() * data['input'].element_size()
            target_size = data['target'][:, stem].numel() * data['target'][:, stem].element_size()
            total_size = input_size + target_size

            # --- Load to GPU/CPU based on priority ---
            inputs = data['input'].unsqueeze(1)
            targets = data['target'][:, stem].unsqueeze(1)

            # Prioritize inputs, then targets
            if available_memory > total_size:
                inputs = inputs.to(device)
                targets = targets.to(device)
            elif available_memory > input_size:
                inputs = inputs.to(device)
                targets = targets.to('cpu')
            else:
                inputs = inputs.to('cpu')
                targets = targets.to('cpu')

            # --- Offloading & Checkpointing ---
            if available_memory < (input_size + target_size):
                model.to('cpu')
                model.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs.to(device))
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]

                loss_g = loss_function_g(outputs.to(device), targets.to(device))
                scaler.scale(loss_g).backward()

                real_labels = torch.ones(inputs.size(0), 1, device=device)
                fake_labels = torch.zeros(inputs.size(0), 1, device=device)
                real_out = discriminator(targets.to(device).clone().detach())
                fake_out = discriminator(outputs.clone().detach())

                loss_d_real = loss_function_d(real_out, real_labels)
                loss_d_fake = loss_function_d(fake_out, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                scaler.scale(loss_d).backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
                    scaler.step(optimizer_g)
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()

                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                    if tensorboard_flag:
                        writer.add_scalar('Loss/Generator', running_loss_g / accumulation_steps, epoch * len(train_loader) + i)
                        writer.add_scalar('Loss/Discriminator', running_loss_d / accumulation_steps, epoch * len(train_loader) + i)
                    running_loss_g = 0.0
                    running_loss_d = 0.0

            del inputs, targets, outputs
            torch.cuda.empty_cache()

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

        # Validation
        model.eval()
        val_loss = 0.0
        sdr_total, sir_total, sar_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if data is None:
                    continue

                logger.debug(f"Validating batch {i+1}/{len(val_loader)}")

                inputs = data['input'].unsqueeze(1).to(device)
                targets = data['target'][:, stem].unsqueeze(1).to(device)
                outputs = model(inputs)
                target_length = targets.size(-1)
                outputs = outputs[..., :target_length]
                loss = loss_function_g(outputs, targets)
                val_loss += loss.item()

                for j in range(outputs.size(0)):
                    sdr, sir, sar = compute_adversarial_loss(targets[j], outputs[j], device)
                    if not torch.isnan(sdr):
                        sdr_total += sdr.mean()
                        sir_total += sir.mean()
                        sar_total += sar.mean()

        # Calculate averages, excluding NaN values
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

    final_model_path = f"{checkpoint_dir}/model_final_stem_{stem}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed for stem {stem}. Final model saved at {final_model_path}")

    if tensorboard_flag:
        writer.close()

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, cache_dir, apply_data_augmentation=False, suppress_messages=False, num_workers=4, device='cpu'):
        # Initialization
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        self.apply_data_augmentation = apply_data_augmentation
        self.suppress_messages = suppress_messages
        self.num_workers = num_workers
        self.device = device  # Add device initialization
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.valid_stems = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

        self.num_processed_stems = num_processed_stems
        self.lock = dataset_lock

        self.mel_spectrogram_params = {
            'sample_rate': 22050,
            'n_fft': self.n_fft,
            'win_length': None,
            'hop_length': self.n_fft // 4,
            'n_mels': self.n_mels,
            'power': 2.0
        }

        self.workers = self._start_workers()
        self.processed_data = []
        self._process_stems()

    def _get_cache_path(self, stem_name, augmentation_flag, noise_flag, noise_amount):
        cache_key = f"{stem_name}_{augmentation_flag}_{noise_flag}_{noise_amount}_{self.mel_spectrogram_params}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()  # Hash for safe filenames
        cache_subdir = "augmented" if augmentation_flag else "original"
        cache_dir = os.path.join(self.cache_dir, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_key + ".pt")

    def _load_from_cache(self, cache_path):
        try:
            logger.info(f"Loading from cache: {cache_path}")
            return torch.load(cache_path)
        except (FileNotFoundError, RuntimeError):  # Handle missing or corrupted cache
            logger.warning(f"Cache not found or corrupted: {cache_path}")
            return None

    def _save_to_cache(self, cache_path, data):
        logger.info(f"Saving to cache: {cache_path}")
        torch.save(data, cache_path)

    def _start_workers(self):
        workers = []
        # Divide the work evenly among workers
        stems_per_worker = len(self.valid_stems) // self.num_workers
        for i in range(self.num_workers):
            start_idx = i * stems_per_worker
            end_idx = start_idx + stems_per_worker if i < self.num_workers - 1 else len(self.valid_stems)
            worker_stems = self.valid_stems[start_idx:end_idx]
            
            # Pass device argument as a string to avoid pickling issues
            p = Process(target=worker, args=(
                self.input_queue, self.output_queue, self.mel_spectrogram_params, 
                self.data_dir, self.target_length, str(self.device), self.apply_data_augmentation,
                worker_stems, self.lock, self.num_processed_stems
            ))

            p.start()
            workers.append(p)
        return workers

    def _process_stems(self):
        """Waits for the workers to finish and stores the processed data."""
        processed_data = []
        while True:
            try:
                stem_name, data = self.output_queue.get(timeout=60)  # Increased timeout
                if data is None:  # Check for None values (possible errors)
                    continue
                logger.info(f"Processed stem: {stem_name}")
                processed_data.append((stem_name, data))
            except queue.Empty:
                with dataset_lock:
                    if num_processed_stems.value >= len(self):
                        logger.info("All stems processed, ending iteration.")
                        break
            except (ConnectionResetError, EOFError) as e:
                logger.error(f"Connection error in _process_stems: {e}")
                self._restart_workers()

        # Sort the processed data by stem name to maintain original order
        processed_data.sort(key=lambda x: x[0])

        # Extract only the preprocessed data (without stem name)
        self.processed_data = [x[1] for x in processed_data]

    def _restart_workers(self):
        # Terminate existing workers
        logger.info("Restarting workers due to connection error.")
        for worker in self.workers:
            worker.terminate()
        # Start new workers
        self.workers = self._start_workers()

    def __len__(self):
        return len(self.valid_stems)

    def __getitem__(self, idx):
        stem_name = self.valid_stems[idx]
        cache_path = self._get_cache_path(stem_name, self.apply_data_augmentation, add_noise, noise_amount)

        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data

        # If not cached, process and save to cache
        logger.info(f"Processing stem: {stem_name}")
        data = self._process_single_stem(stem_name)
        self._save_to_cache(cache_path, data)
        return data

    def _process_single_stem(self, stem_name):
        file_path = os.path.join(self.data_dir, stem_name)
        logger.info(f"Loading and preprocessing: {file_path}")
        mel_spectrogram = T.MelSpectrogram(**self.mel_spectrogram_params).to(self.device)
        return load_and_preprocess(file_path, mel_spectrogram, self.target_length, self.apply_data_augmentation, self.device)

# Example usage
if __name__ == "__main__":
    data_dir = r"K:\KAN-Stem DataSet\ProcessedDataset"
    val_dir = r"K:\KAN-Stem DataSet\ValidationDataset"
    n_mels = 64
    target_length = 128
    n_fft = 1024
    cache_dir = r"K:\KAN-Stem DataSet\Cache"
    apply_data_augmentation = True
    suppress_warnings = False
    num_workers = 4
    batch_size = 8
    num_epochs = 10
    learning_rate_g = 0.001
    learning_rate_d = 0.001
    use_cuda = True
    checkpoint_dir = r"K:\KAN-Stem DataSet\Checkpoints"
    save_interval = 5
    accumulation_steps = 4
    num_stems = 1
    loss_function_str_g = "MSELoss"
    loss_function_str_d = "BCEWithLogitsLoss"
    optimizer_name_g = "Adam"
    optimizer_name_d = "Adam"
    perceptual_loss_flag = False
    clip_value = 1.0
    scheduler_step_size = 5
    scheduler_gamma = 0.5
    tensorboard_flag = True
    add_noise = False
    noise_amount = 0.01
    early_stopping_patience = 3
    weight_decay = 0.0001
    suppress_reading_messages = True
    use_cpu_for_prep = True

    start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep)
