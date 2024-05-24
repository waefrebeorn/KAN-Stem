import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging

def analyze_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    if waveform.abs().max() == 0:  # Check for silence
        return sample_rate, duration, 0, True  # Indicate that the audio is silent
    return sample_rate, duration, waveform.size(1), False  # Return sequence length instead of None

def detect_parameters(data_dir, default_n_mels=64, default_n_fft=1024):
    sample_rates = []
    durations = []

    # Print the contents of the data directory
    print(f"Contents of the data directory ({data_dir}): {os.listdir(data_dir)}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):  # Ensure only .wav files are processed
            file_path = os.path.join(data_dir, file_name)
            sample_rate, duration, sequence_length, is_silent = analyze_audio(file_path)
            if is_silent:
                print(f"Skipping silent file: {file_name}")
                continue  # Skip silent files
            sample_rates.append(sample_rate)
            durations.append(duration)

    print(f"Found {len(sample_rates)} valid audio files")
    
    if not sample_rates or not durations:
        raise ValueError("No valid audio files found in the dataset")

    avg_sample_rate = sum(sample_rates) / len(sample_rates)
    avg_duration = sum(durations) / len(durations)

    # Set n_mels and n_fft based on average sample rate and duration
    n_mels = min(default_n_mels, int(avg_sample_rate / 100))
    n_fft = min(default_n_fft, int(avg_sample_rate * 0.025))  # 25 ms window

    return int(avg_sample_rate), n_mels, n_fft

def _waveform_to_spectrogram_librosa(waveform, sr, n_mels=64, n_fft=1024):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB
class KAN(nn.Module):
    def __init__(self, in_features):
        super(KAN, self).__init__()
        self.layer1 = nn.Linear(in_features, 512)
        self.fc = nn.Linear(512, in_features)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger):
    """Trains the KAN model for one epoch."""
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs = data.to(device)

        # Ensure correct dimensions
        if inputs.dim() == 1:  # Flattened input
            n_mels = int(inputs.size(0) ** 0.5)  # Assuming square input
            seq_len = int(inputs.size(0) / n_mels)
            inputs = inputs.view(1, n_mels, seq_len)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

        # Save checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0 and i == len(train_loader) - 1:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

    writer.close()
    logger.info('Epoch Made')
    return "Epoch Made"
def start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)

    dataset = CustomDataset(data_dir, n_mels=n_mels, target_length=256, n_fft=n_fft)
    if len(dataset) == 0:
        raise ValueError("No valid audio files found in the dataset after filtering.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False)

    # Calculate in_features based on n_mels and target_length
    target_length = 256
    in_features = int(n_mels * target_length)

    model = KAN(in_features=in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger)

    writer.close()
    logger.info('Training Finished')

    return "Training Finished"
def perform_separation(checkpoint_path, file_path, n_mels=64, target_length=256, n_fft=1024):
    """Performs stem separation on a given audio file using the loaded model."""

    # Load checkpoint and get in_features
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # Handle both new and old checkpoint formats
    in_features = model_state_dict['layer1.weight'].shape[1]

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.abs().max() == 0:
        raise ValueError("The provided audio file is silent and cannot be processed.")

    if waveform.shape[0] == 1:  # Handle mono audio by duplicating to stereo
        waveform = torch.cat([waveform, waveform], dim=0)

    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft
    )(waveform)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    # Handle batch dimension if present (e.g., from multiple files)
    if spectrogram.dim() > 2:
        spectrogram = spectrogram[0]  # Take the first item in the batch

    # Pad or truncate to target length
    if spectrogram.size(-1) > target_length:
        spectrogram = spectrogram[:, :target_length]
    else:
        padding = target_length - spectrogram.size(-1)
        spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))

    # Correctly flatten the spectrogram (2D -> 1D)
    spectrogram = spectrogram.flatten().unsqueeze(0)

    # Load the model
    model = load_model(checkpoint_path, in_features)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    spectrogram = spectrogram.to(device)

    # Separate stems
    with torch.no_grad():
        separated_spectrogram = model(spectrogram)

    # Reshape and convert back to waveform
    separated_spectrogram = separated_spectrogram.view(n_mels, target_length).cpu()
    
    # Adjust n_fft for GriffinLim
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    griffin_lim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=32)
    
    separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrogram))

    return separated_waveform

def load_model(checkpoint_path, in_features):
    """Loads the model from a checkpoint."""
    model = KAN(in_features=in_features)
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def test_cuda():
    return torch.cuda.is_available()

def separate_stems(model, file_path, output_path, n_mels=64, n_fft=1024, target_length=256, device='cpu'):
    sample_rate, duration, sequence_length, is_silent = analyze_audio(file_path)
    if is_silent:
        raise ValueError("The provided audio file is silent and cannot be processed.")
    
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.numpy().flatten()
    spectrogram = _waveform_to_spectrogram_librosa(waveform, sample_rate, n_mels=n_mels, n_fft=n_fft)
    spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)  # Normalize spectrogram

    # Truncate or pad the spectrogram to match the training target length
    if spectrogram.shape[1] > target_length:
        spectrogram = spectrogram[:, :target_length]
    else:
        padding = target_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')

    spectrogram = torch.tensor(spectrogram).unsqueeze(0).to(device)  # Add batch dimension and move to device
    spectrogram = spectrogram.view(1, n_mels * target_length)  # Reshape for model input

    with torch.no_grad():
        separated_spectrogram = model(spectrogram)

    separated_spectrogram = separated_spectrogram.view(n_mels, target_length).cpu()  # Reshape back to spectrogram dimensions and move to CPU

    # Convert the spectrogram back to waveform using Griffin-Lim algorithm
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    griffin_lim_transform = torchaudio.transforms.GriffinLim(n_iter=32)
    separated_waveform = griffin_lim_transform(inverse_mel_transform(separated_spectrogram))
    torchaudio.save(output_path, separated_waveform, sample_rate)
    return output_path

if __name__ == "__main__":
    data_dir = "your_dataset_directory"
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    use_cuda = True
    checkpoint_dir = "./checkpoints"
    save_interval = 1

    start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval)
