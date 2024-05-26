import os
import psutil
import gc

# Function to print current memory usage
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{step}: Memory usage: {mem_info.rss / 1024**2:.2f} MB")

# Print memory usage immediately after importing basic modules
print_memory_usage("After importing os, psutil, gc")

import torch  # Import torch first
print_memory_usage("After importing torch")

import torchaudio 
print_memory_usage("After importing torchaudio")

from torch.utils.data import DataLoader, Dataset
print_memory_usage("After importing DataLoader, Dataset")

import torch.nn as nn
print_memory_usage("After importing torch.nn")
import torch.optim as optim
import argparse
import psutil
import gc
import tracemalloc

# Function to print current memory usage
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{step}: Memory usage: {mem_info.rss / 1024**2:.2f} MB")

class CustomDataset(Dataset):
    def __init__(self, data_dir, chunk_size, hop_length, n_fft):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.input_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
        self.target_files = {i: [] for i in range(7)}
        target_prefixes = ['target_bass_', 'target_drums_', 'target_guitar_', 'target_keys_', 'target_noise_', 'target_other_', 'target_vocals_']
        for i, prefix in enumerate(target_prefixes):
            self.target_files[i] = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])

        self.num_stems = len(target_prefixes)

    def __len__(self):
        return len(self.input_files) * self.num_stems

    def __getitem__(self, index):
        file_index = index // self.num_stems
        stem_index = index % self.num_stems

        input_file = os.path.join(self.data_dir, self.input_files[file_index])
        target_file = os.path.join(self.data_dir, self.target_files[stem_index][file_index])

        chunk_start = file_index * self.hop_length
        chunk_end = chunk_start + self.chunk_size

        with torch.no_grad():
            input_waveform, sample_rate = torchaudio.load(
                input_file,
                num_frames=self.chunk_size,
                frame_offset=chunk_start
            )
            input_waveform = input_waveform.to(torch.float16)
            input_spectrogram = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(input_waveform)

            target_waveform, _ = torchaudio.load(
                target_file,
                num_frames=self.chunk_size,
                frame_offset=chunk_start
            )
            target_waveform = target_waveform.to(torch.float16)
            target_spectrogram = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(target_waveform)

        return input_spectrogram.to(torch.float16), target_spectrogram.to(torch.float16), stem_index



    def chunk_spectrogram(self, spectrogram, is_input):
        num_chunks = (spectrogram.shape[-1] - self.chunk_size) // self.hop_length + 1
        chunks = []
        for i in range(num_chunks):
            start = i * self.hop_length
            end = start + self.chunk_size
            chunk = spectrogram[..., start:end]
            chunks.append(chunk)
        
        if is_input:
            chunks = torch.stack(chunks, dim=2)  # Stack along the third dimension for input spectrograms
        else:
            chunks = torch.stack(chunks, dim=2)  # Stack along the third dimension for target spectrograms
            chunks = chunks.unsqueeze(1)  # Add a dummy batch dimension for target spectrograms
            chunks = chunks.permute(1, 0, 2, 3, 4, 5)  # Permute dimensions to match the expected target shape
        
        print_memory_usage("After chunking spectrogram")
        return chunks
    
    def clear_memory(self, *args):
        for arg in args:
            del arg
        torch.cuda.empty_cache()
        gc.collect()

def load_data(dataset_path, batch_size, chunk_size, hop_length, n_fft):
    print(f"Loading data from {dataset_path} with batch size {batch_size}")
    train_dataset = CustomDataset(dataset_path, chunk_size, hop_length, n_fft)

    total_chunks = len(train_dataset.input_files) * (train_dataset.chunk_samples - train_dataset.chunk_size) // train_dataset.hop_length + 1
    total_chunks *= 7
    print(f"Total number of chunks: {total_chunks}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader

class SplineActivation(nn.Module):
    def __init__(self, num_params=1, num_knots=10):
        super(SplineActivation, self).__init__()
        self.num_params = num_params
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots))
        self.coeffs = nn.Parameter(torch.ones(num_knots, num_params))

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x = x.unsqueeze(-1).expand(-1, -1, self.num_knots)
        x = torch.bmm(x, self.coeffs.unsqueeze(0).expand(x.size(0), -1, -1))
        x = x.view(*x_shape[:-1], -1)
        return x.sum(dim=-1)

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super(KANLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = SplineActivation(num_params=output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class KANModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, degree=3, num_layers=8):
        super(KANModel, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for _ in range(num_layers - 1):
            self.layers.append(KANLayer(current_size, hidden_size, degree))
            current_size = hidden_size
        self.layers.append(KANLayer(current_size, 7 * hidden_size, degree))  # Output 7 stems with hidden_size each
        print("KANModel initialization completed.")

    def forward(self, x):
        try:
            print(f"Input shape: {x.shape}")
            # Ensure the input tensor has the correct shape for the linear layers
            batch_size, num_channels, num_frequencies, num_timesteps, chunk_size = x.shape
            x = x.view(batch_size, num_channels * num_frequencies * num_timesteps * chunk_size)

            input_size = self.layers[0].linear.in_features
            if x.shape[1] != input_size:
                print(f"Input tensor shape mismatch: Expected {input_size} features, but got {x.shape[1]}")
                raise ValueError("Input tensor shape mismatch")

            for layer in self.layers:
                x = layer(x)
                print(f"Layer output shape: {x.shape}")

            # Reshape the output tensor to match the target shape
            x = x.view(batch_size, 7, num_channels, num_frequencies, num_timesteps, chunk_size)
            return x
        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise e

def train(model, train_loader, criterion, optimizer, device):
    stem_losses = [0.0 for _ in range(7)]
    for batch_idx, (inputs, targets, stem_indices) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        for i, stem_idx in enumerate(stem_indices):
            stem_losses[stem_idx] += loss[i].item()  # Assumes batch size of 1

    for i, stem_loss in enumerate(stem_losses):
        print(f"Stem {i}: Average Loss: {stem_loss / len(train_loader):.4f}")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def main():
    tracemalloc.start()

    parser = argparse.ArgumentParser(description='KAN Training Script')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--chunk_size', type=int, default=256, help='Chunk size for spectrogram')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length for spectrogram')
    parser.add_argument('--n_fft', type=int, default=512, help='Number of FFT points')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for KAN layers')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print memory usage before initializing the model
    print_memory_usage("Before model initialization")

    input_size = args.chunk_size * (args.n_fft // 2 + 1) * 483 * 1
    model = KANModel(input_size=input_size, hidden_size=args.hidden_size).to(device)

    # Print memory usage after initializing the model
    print_memory_usage("After model initialization")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}...")
        try:
            train_loss = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{args.num_epochs} completed. Training Loss: {train_loss:.4f}")

            # Save checkpoint
            checkpoint_path = f"checkpoints/kan_model_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Clear memory after each epoch
            torch.cuda.empty_cache()
            gc.collect()
            print_memory_usage(f"After epoch {epoch+1}")
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            raise e

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 memory-consuming lines ]")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == "__main__":
    main()
