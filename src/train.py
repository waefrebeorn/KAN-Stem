import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import BSpline
import argparse
import time
import soundfile as sf
import psutil

# Define learnable activation function
class SplineActivation(nn.Module):
    def __init__(self, num_params=1, num_knots=10):
        super(SplineActivation, self).__init__()
        self.num_params = num_params
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots))
        self.coeffs = nn.Parameter(torch.ones(num_knots, num_params))

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, self.num_knots)
        x = BSpline(self.knots, self.coeffs, 3)(x)
        return x.sum(dim=-1)

# Define KANLayer
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super(KANLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = SplineActivation(num_params=output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Define KANModel
class KANModel(nn.Module):
    def __init__(self, input_size, num_layers=3, degree=3):
        super(KANModel, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for _ in range(num_layers - 1):
            self.layers.append(KANLayer(current_size, current_size * 2 + 1, degree))
            current_size = current_size * 2 + 1
        self.layers.append(KANLayer(current_size, 1, degree))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)  # Output shape: (batch_size, 1)

# Custom dataset class
class AudioDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        input_file, target_files = self.data_files[idx]
        input_waveform, sample_rate = sf.read(input_file, dtype='float32')
        input_waveform = torch.tensor(input_waveform).float()

        target_waveforms = []
        for target_file in target_files:
            target_waveform, _ = sf.read(target_file, dtype='float32')
            target_waveforms.append(torch.tensor(target_waveform).float())

        return input_waveform, torch.stack(target_waveforms)

def load_dataset(dataset_path, max_examples):
    data_files = []
    for i in range(1, max_examples + 1):
        input_file = os.path.join(dataset_path, f'input_{i}.wav')
        target_files = [
            os.path.join(dataset_path, f'target_vocals_{i}.wav'),
            os.path.join(dataset_path, f'target_drums_{i}.wav'),
            os.path.join(dataset_path, f'target_bass_{i}.wav'),
            os.path.join(dataset_path, f'target_guitar_{i}.wav'),
            os.path.join(dataset_path, f'target_keys_{i}.wav'),
        ]
        data_files.append((input_file, target_files))

    chunk_size = 1  # Very small chunk size to manage memory usage
    data_files = [data_files[i:i + chunk_size] for i in range(0, len(data_files), chunk_size)]

    return data_files

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"Batch {batch_idx}, Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Outputs shape: {outputs.shape}")
            loss = criterion(outputs, targets)
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

def main(args):
    print(f"Loading dataset from {args.dataset}")
    max_examples = 5  # Very small number to ensure memory constraints
    data_chunks = load_dataset(args.dataset, max_examples)
    if not data_chunks:
        print("No data found. Exiting...")
        return

    print(f"Loaded {len(data_chunks)} data chunks.")
    model = KANModel(input_size=44100, num_layers=args.num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device('cpu')
    print(f"Using device: {device}")

    for chunk_index, data_files in enumerate(data_chunks):
        print(f"Processing chunk {chunk_index + 1}/{len(data_chunks)}")
        dataset = AudioDataset(data_files)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        model = train_model(model, train_loader, criterion, optimizer, args.num_epochs, device)

        # Save the trained model after each chunk
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), os.path.join('checkpoints', f'model_chunk_{chunk_index + 1}_epoch_{args.num_epochs}.ckpt'))

        # Debug: show current memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Current memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KANModel")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of KAN layers')
    
    args = parser.parse_args()
    main(args)
