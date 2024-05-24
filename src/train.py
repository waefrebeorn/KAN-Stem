import os
import torch
import torchaudio
import librosa
import numpy as np
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import logging
import tempfile
import soundfile as sf
from model import KANWithDepthwiseConv  # Import the model from model.py


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

# Convert waveform to spectrogram using librosa
def _waveform_to_spectrogram_librosa(waveform, sr, n_mels=64, n_fft=1024):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=self.n_mels, n_fft=self.n_fft
        )(waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        return spectrogram  # Shape [channels, height, width]

# Define depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Define the KAN model with depthwise convolution
class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length):
        super(KANWithDepthwiseConv, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.fc = nn.Conv2d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = checkpoint.checkpoint(self._forward_impl, x, use_reentrant=True)
        return x

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    scaler = amp.GradScaler()

    optimizer.zero_grad()
    for i, data in enumerate(train_loader, 0):
        inputs = data.to(device)
        print(f"Input shape before unsqueeze: {inputs.shape}")  # Debugging line

        # Ensure inputs require gradients
        inputs.requires_grad_(True)

        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        if i % (10 * accumulation_steps) == (10 * accumulation_steps - 1):
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    logger.info(torch.cuda.memory_summary(device))

    if (epoch + 1) % save_interval == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    writer.close()
    logger.info('Epoch Made')
    return "Epoch Made"

# Start training
def start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps=4):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)

    dataset = CustomDataset(data_dir, n_mels=n_mels, target_length=256, n_fft=n_fft)
    if len(dataset) == 0:
        raise ValueError("No valid audio files found in the dataset after filtering.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False)

    target_length = 256
    in_channels = 1  # Single channel for 2D spectrograms
    out_channels = 64  # Adjust the number of output channels as needed

    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger, accumulation_steps)

    writer.close()
    logger.info('Training Finished')

    return "Training Finished"

# Perform stem separation
def perform_separation(checkpoint_path, file_path, n_mels=64, target_length=256, n_fft=1024):
    """Performs stem separation on a given audio file using the loaded model."""

    # Load checkpoint and get in_features
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    in_channels = model_state_dict['conv1.depthwise.weight'].shape[1]
    out_channels = model_state_dict['conv1.pointwise.weight'].shape[0]

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.abs().max() == 0:
        raise ValueError("The provided audio file is silent and cannot be processed.")

    if waveform.shape[0] == 1:  # Handle mono audio by duplicating to stereo
        waveform = torch.cat([waveform, waveform], dim=0)

    # Load the model
    model = load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Process the audio in chunks
    chunk_size = sample_rate * target_length // n_mels
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
    separated_audio = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, waveform.shape[1])
        chunk_waveform = waveform[:, start_idx:end_idx]

        if chunk_waveform.shape[1] < chunk_size:
            padding = chunk_size - chunk_waveform.shape[1]
            chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding))

        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft
        )(chunk_waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        spectrogram = spectrogram.unsqueeze(0).to(device)

        with torch.no_grad():
            separated_spectrogram = model(spectrogram)

        separated_spectrogram = separated_spectrogram.squeeze().cpu()
        inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
        griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)

        separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrogram))
        separated_audio.append(separated_waveform[:, :chunk_waveform.shape[1]])

    separated_audio = torch.cat(separated_audio, dim=1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, separated_audio.numpy().T, sample_rate)
        tmpfile_path = tmpfile.name

    return tmpfile_path

# Test CUDA availability
def test_cuda():
    return torch.cuda.is_available()

if __name__ == "__main__":
    data_dir = "your_dataset_directory"
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    use_cuda = True
    checkpoint_dir = "./checkpoints"
    save_interval = 1

    start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval)
