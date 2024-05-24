import gradio as gr
import os
import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging

def analyze_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    max_amplitude = waveform.abs().max()
    print(f"Analyzing {file_path}: Sample Rate = {sample_rate}, Duration = {duration}, Max Amplitude = {max_amplitude}")
    if max_amplitude == 0:  # Check for silence
        return sample_rate, duration, True  # Indicate that the audio is silent
    return sample_rate, duration, False

def detect_parameters(data_dir, default_n_mels=128, default_n_fft=512):
    sample_rates = []
    durations = []

    # Print the contents of the data directory
    print(f"Contents of the data directory ({data_dir}): {os.listdir(data_dir)}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):  # Ensure only .wav files are processed
            file_path = os.path.join(data_dir, file_name)
            sample_rate, duration, is_silent = analyze_audio(file_path)
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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_mels=128, target_length=256):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav') and not analyze_audio(os.path.join(data_dir, f))[2]]
        self.n_mels = n_mels
        self.target_length = target_length

        # Print the number of files in the dataset
        print(f"Number of files in the dataset: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        spectrogram = self._waveform_to_spectrogram(waveform)

        # Truncate or pad spectrogram to target length
        if spectrogram.size(-1) > self.target_length:
            spectrogram = spectrogram[:, :, :self.target_length]
        else:
            padding = self.target_length - spectrogram.size(-1)
            spectrogram = nn.functional.pad(spectrogram, (0, padding))

        # Add a channel dimension (Since MelSpectrogram returns a single-channel spectrogram)
        spectrogram = spectrogram.unsqueeze(0)  # Adding one channel dimension

        return spectrogram

    def _waveform_to_spectrogram(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)(waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
        return spectrogram

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.inorm = nn.InstanceNorm1d(out_features, affine=True)

    def forward(self, x):
        x = self.fc(x)
        if x.size(0) > 1:  # Use batch normalization if batch size is greater than 1
            x = self.bn(x)
        else:  # Use instance normalization if batch size is 1
            x = x.view(1, -1, x.size(-1))  # Reshape for instance normalization
            x = self.inorm(x)
            x = x.view(1, -1)  # Reshape back to original
        return nn.functional.relu(x)

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
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs = data.to(device)  # Assuming audio data is the first element

        # Squeeze unnecessary dimensions
        inputs = inputs.squeeze()

        # Ensure the input tensor has the correct dimensions
        if inputs.ndimension() == 2:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if necessary
        elif inputs.ndimension() == 4:
            batch_size, c, n_mels, seq_len = inputs.shape
            inputs = inputs.view(batch_size, n_mels, seq_len)
        elif inputs.ndimension() == 5:
            batch_size, c1, c2, n_mels, seq_len = inputs.shape
            inputs = inputs.view(batch_size, n_mels, seq_len)
        elif inputs.ndimension() != 3:
            print("Unexpected input dimensions after squeezing. Expected 3D tensor, got: ", inputs.shape)
            continue

        batch_size, n_mels, seq_len = inputs.shape
        in_features = int(n_mels * seq_len)
        inputs = inputs.view(batch_size, in_features)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

        # Save checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    writer.close()
    logger.info('Training Finished')
    return "Training Finished"

def start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)

    dataset = CustomDataset(data_dir, n_mels=n_mels, target_length=256)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False)

    # Calculate in_features based on n_mels and target_length
    target_length = 256  # Example value, adjust as needed
    in_features = int(n_mels * target_length)

    # Initialize the model with in_features
    model = KAN(in_features=in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, logger)

    writer.close()
    logger.info('Training Finished')

    return "Training Finished"

def test_cuda():
    return torch.cuda.is_available()

def load_model(checkpoint_path, in_features):
    model = KAN(in_features=int(in_features))
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def separate_stems(model, file_path, output_path):
    sample_rate, n_fft = analyze_audio(file_path)[:2]
    waveform, sample_rate = torchaudio.load(file_path)
    spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=128, n_fft=n_fft)(waveform)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
    spectrogram = spectrogram[:, :, :256] if spectrogram.size(-1) > 256 else nn.functional.pad(spectrogram, (0, 256 - spectrogram.size(-1)))
    spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        separated_spectrogram = model(spectrogram)
    separated_spectrogram = separated_spectrogram.squeeze(0)  # Remove batch dimension
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=128)
    griffin_lim = torchaudio.transforms.GriffinLim(n_iter=32)
    separated_waveform = griffin_lim(inverse_mel(separated_spectrogram))
    torchaudio.save(output_path, separated_waveform, sample_rate)
    return output_path

def perform_separation(checkpoint_path, file_path):
    sample_rate, n_mels, seq_len = analyze_audio(file_path)
    in_features = int(n_mels * seq_len)  # Calculate in_features based on actual sequence length
    model = load_model(checkpoint_path, in_features=in_features)
    output_path = "separated_output.wav"
    separated_file_path = separate_stems(model, file_path, output_path)
    return separated_file_path

with gr.Blocks() as demo:
    with gr.Tab("Train Model"):
        gr.Markdown("## CUDA Test and KAN Training")
        
        btn_cuda = gr.Button("Check CUDA Availability")
        output_cuda = gr.Textbox()
        btn_cuda.click(test_cuda, outputs=output_cuda)
        
        gr.Markdown("### Start Training")
        data_dir = gr.Textbox(label="Dataset Directory", value="/path/to/your/dataset")
        batch_size = gr.Number(label="Batch Size", value=4)
        num_epochs = gr.Number(label="Number of Epochs", value=10)
        learning_rate = gr.Number(label="Learning Rate", value=0.001)
        use_cuda = gr.Checkbox(label="Use CUDA", value=True)
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
        save_interval = gr.Number(label="Checkpoint Save Interval (epochs)", value=1)
        
        btn_train = gr.Button("Start Training")
        output_train = gr.Textbox()
        
        btn_train.click(start_training, inputs=[data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval], outputs=output_train)
    
    with gr.Tab("Separate Stems"):
        gr.Markdown("## Stem Separation using KAN Model")
        
        checkpoint_path = gr.Textbox(label="Checkpoint Path", value="./checkpoints/checkpoint_epoch_10.pth")
        file_path = gr.Textbox(label="Audio File Path", value="/path/to/your/audio/file.wav")
        
        btn_separate = gr.Button("Separate Stems")
        output_separation = gr.Audio(label="Separated Audio")
        
        btn_separate.click(perform_separation, inputs=[checkpoint_path, file_path], outputs=output_separation)

demo.launch(server_name="127.0.0.1", server_port=7861)