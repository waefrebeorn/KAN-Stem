import os
import torch
import torchaudio
import numpy as np
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import logging
import matplotlib.pyplot as plt
import soundfile as sf
from model import KANWithDepthwiseConv, KANDiscriminator, load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hook to log when a tensor is allocated
def allocation_hook(tensor):
    logger.info(f"Allocated tensor of size {tensor.size()} with memory {tensor.element_size() * tensor.nelement() / 1024 ** 2:.2f} MB")

# Register hooks
original_new = torch.Tensor.__new__
def new_tensor(cls, *args, **kwargs):
    tensor = original_new(cls, *args, **kwargs)
    allocation_hook(tensor)
    return tensor

torch.Tensor.__new__ = new_tensor

# Analyze audio file
def analyze_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.shape[1] / sample_rate
    is_silent = waveform.abs().max() == 0
    sequence_length = waveform.shape[1]
    return sample_rate, duration, sequence_length, is_silent

# Detect parameters for dataset
def detect_parameters(data_dir, default_n_mels=64, default_n_fft=1024):
    sample_rates = []
    durations = []

    print(f"Contents of the data directory ({data_dir}): {os.listdir(data_dir)}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            sample_rate, duration, sequence_length, is_silent = analyze_audio(file_path)
            if is_silent:
                print(f"Skipping silent file: {file_name}")
                continue
            sample_rates.append(sample_rate)
            durations.append(duration)

    print(f"Found {len(sample_rates)} valid audio files")
    
    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    avg_sample_rate = sum(sample_rates) / len(sample_rates)
    avg_duration = sum(durations) / len(durations)

    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))

    return int(avg_sample_rate), n_mels, n_fft

# Custom dataset class
class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, num_stems):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.num_stems = num_stems  # Number of stems to separate
        self.file_list = []

        input_files = [f for f in os.listdir(data_dir) if f.startswith("input_") and f.endswith(".wav")]

        for f in input_files:
            mix_path = os.path.join(data_dir, f)
            stem_paths = [os.path.join(data_dir, f.replace("input_", f"target_{stem}_")) for stem in ["bass", "drums", "guitar", "keys", "noise", "other", "vocals"]]  # Assuming these stems
            if all(os.path.exists(path) for path in stem_paths):  # Ensure all stems are present
                self.file_list.append((mix_path, stem_paths))

        total_files = len(input_files) + len(input_files) * self.num_stems
        logger.info(f"Total number of files considered (inputs + stems): {total_files}")
        logger.info(f"Number of valid file sets (input + corresponding stems): {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mix_path, stem_paths = self.file_list[idx]

        # Load and preprocess the mix audio
        mix_waveform, sample_rate = torchaudio.load(mix_path)
        mix_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=self.n_mels, n_fft=self.n_fft
        )(mix_waveform)
        mix_spectrogram = (mix_spectrogram - mix_spectrogram.mean()) / mix_spectrogram.std()

        # Load and preprocess each stem
        stem_spectrograms = []
        for stem_path in stem_paths:
            stem_waveform, _ = torchaudio.load(stem_path)

            if stem_waveform.shape[0] != mix_waveform.shape[0]:  
                stem_waveform = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)(stem_waveform)  
            
            # Ensure the stem has the same number of samples as the mix
            if stem_waveform.shape[1] < mix_waveform.shape[1]:
                # Pad if the stem is shorter
                stem_waveform = F.pad(stem_waveform, (0, mix_waveform.shape[1] - stem_waveform.shape[1]))
            elif stem_waveform.shape[1] > mix_waveform.shape[1]:
                # Truncate if the stem is longer
                stem_waveform = stem_waveform[:, :mix_waveform.shape[1]]

            stem_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=self.n_mels, n_fft=self.n_fft
            )(stem_waveform)
            stem_spectrogram = (stem_spectrogram - stem_spectrogram.mean()) / stem_spectrogram.std()
            stem_spectrograms.append(stem_spectrogram)

        # Stack the stem spectrograms
        target_spectrogram = torch.stack(stem_spectrograms, dim=0)  # Shape: [num_stems, channels, n_mels, target_length]

        return mix_spectrogram, target_spectrogram

# Training function
def chunked_log_params(logger, params, chunk_size=100):
    """Log parameters in chunks to avoid hitting the enter key too much."""
    chunk = []
    for name, param in params:
        chunk.append((name, param.data))
        if len(chunk) >= chunk_size:
            logger.info(chunk)
            chunk = []
    if chunk:
        logger.info(chunk)

def save_params_chart(params, epoch, checkpoint_dir):
    """Save parameter values as a chart."""
    plt.figure(figsize=(12, 6))
    for name, param in params:
        plt.plot(param.detach().cpu().numpy().flatten(), label=name)
    plt.legend()
    plt.title(f'Model Parameters at Epoch {epoch}')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.grid(True)
    chart_path = os.path.join(checkpoint_dir, f'params_epoch_{epoch}.png')
    plt.savefig(chart_path)
    plt.close()

def train(model, discriminator, train_loader, criterion, optimizer_g, optimizer_d, scheduler_g, scheduler_d, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger, accumulation_steps=4, verbose=False):
    model.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0
    scaler = amp.GradScaler()

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    for i, (inputs, targets) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs.requires_grad = True

        with amp.autocast():
            outputs = model(inputs)
            targets = targets.view_as(outputs)
            loss_g = criterion(outputs, targets)
            loss_g = loss_g / accumulation_steps

        scaler.scale(loss_g).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad()
            scheduler_g.step()

        running_loss_g += loss_g.item() * accumulation_steps

        real_spectrograms = targets.view(targets.size(0), -1, targets.size(2), targets.size(3))
        fake_spectrograms = outputs.detach().view(outputs.size(0), -1, outputs.size(2), outputs.size(3))

        with amp.autocast():
            fake_labels = discriminator(fake_spectrograms)
            real_labels = discriminator(real_spectrograms)
            loss_d = (fake_labels - 1).pow(2).mean() + real_labels.pow(2).mean()
            loss_d = loss_d / accumulation_steps

        scaler.scale(loss_d).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer_d)
            scaler.update()
            optimizer_d.zero_grad()
            scheduler_d.step()

        running_loss_d += loss_d.item() * accumulation_steps

        if i % (50 * accumulation_steps) == (50 * accumulation_steps - 1) and verbose:
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss G: {running_loss_g / 50:.4f}, Loss D: {running_loss_d / 50:.4f}')
            writer.add_scalar('Loss/train_g', running_loss_g / 50, epoch * len(train_loader) + i)
            writer.add_scalar('Loss/train_d', running_loss_d / 50, epoch * len(train_loader) + i)
            running_loss_g = 0.0
            running_loss_d = 0.0

    logger.info(torch.cuda.memory_summary(device))

    if (epoch + 1) % save_interval == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')

        model_state_dict = model.state_dict()
        discriminator_state_dict = discriminator.state_dict()
        optimizer_g_state_dict = optimizer_g.state_dict()
        optimizer_d_state_dict = optimizer_d.state_dict()

        # Log parameter values in chunks
        model_params = list(model.named_parameters())
        discriminator_params = list(discriminator.named_parameters())
        if verbose:
            chunked_log_params(logger, model_params)
            chunked_log_params(logger, discriminator_params)

        # Save parameter values as a chart
        if verbose:
            save_params_chart(model_params, epoch + 1, checkpoint_dir)
            save_params_chart(discriminator_params, epoch + 1, checkpoint_dir)

        torch.save({
            'model_state_dict': model_state_dict,
            'discriminator_state_dict': discriminator_state_dict,
            'optimizer_g_state_dict': optimizer_g_state_dict,
            'optimizer_d_state_dict': optimizer_d_state_dict
        }, checkpoint_path)

        logger.info(f"Checkpoint saved at {checkpoint_path}")

    writer.close()
    logger.info('Epoch Made')
    return "Epoch Made"

# Dummy implementation for the functions used
def create_logger(name):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

# Dummy implementation for the training function call
if __name__ == "__main__":
    logger = create_logger('train')
    model = KANWithDepthwiseConv(7, 64, 64, 256, 7)
    discriminator = KANDiscriminator(7, 64, 64, 256, 7)
    optimizer_g = torch.optim.Adam(model.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1)
    writer = SummaryWriter()
    train_loader = [([torch.randn(7, 64, 64)], [torch.randn(7, 64, 64)])] * 10  # Dummy data loader
    train(model, discriminator, train_loader, nn.MSELoss(), optimizer_g, optimizer_d, scheduler_g, scheduler_d, 'cuda', 0, 10, writer, './checkpoints', 1, logger, verbose=False)
    
# Start training
def start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps=4, num_stems=7):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)

    dataset = StemSeparationDataset(data_dir, n_mels=n_mels, target_length=256, n_fft=n_fft, num_stems=num_stems)
    logger.info(f"Number of valid file sets (input + corresponding stems): {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("No valid audio files found in the dataset after filtering.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False)

    target_length = 256
    in_channels = 1  # Adjusted to handle single-channel input
    out_channels = 64  # Adjust the number of output channels as needed

    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems=num_stems).to(device)
    discriminator = KANDiscriminator(in_channels * num_stems, out_channels, n_mels, target_length, num_stems=num_stems).to(device)
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=0.9)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=0.9)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        train(model, discriminator, train_loader, criterion, optimizer_g, optimizer_d, scheduler_g, scheduler_d, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger, accumulation_steps, verbose=False)

    writer.close()
    logger.info('Training Finished')

    return "Training Finished"

# Test CUDA availability
def test_cuda():
    return torch.cuda.is_available()

def verify_dataset_loading(dataset):
    for i in range(len(dataset)):
        mix_spectrogram, target_spectrogram = dataset[i]
        print(f"Mix Spectrogram Shape: {mix_spectrogram.shape}")
        print(f"Target Spectrogram Shape: {target_spectrogram.shape}")
