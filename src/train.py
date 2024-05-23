import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import argparse

class CustomDataset(Dataset):
    def __init__(self, data_dir, chunk_size, hop_length, n_fft):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Identify input and target files
        self.input_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
        self.target_files = {i: [] for i in range(7)}  # Assuming 7 target categories: bass, drums, guitar, keys, noise, other, vocals

        target_prefixes = ['target_bass_', 'target_drums_', 'target_guitar_', 'target_keys_', 'target_noise_', 'target_other_', 'target_vocals_']
        for i, prefix in enumerate(target_prefixes):
            self.target_files[i] = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])

        print(f"Initialized CustomDataset with {len(self.input_files)} examples")
        print(f"Input files found: {self.input_files}")
        print(f"Target files found: {self.target_files}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file = os.path.join(self.data_dir, self.input_files[index])
        target_files = [os.path.join(self.data_dir, self.target_files[i][index]) for i in range(7)]

        print(f"Loading input file: {input_file}")
        input_waveform, sample_rate = torchaudio.load(input_file)
        print(f"Loaded input file with sample rate: {sample_rate} and shape: {input_waveform.shape}")
        input_spectrogram = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(input_waveform)
        input_chunks = self.chunk_spectrogram(input_spectrogram, is_input=True)
        print(f"Input spectrogram shape: {input_spectrogram.shape}, Input chunks shape: {input_chunks.shape}")

        target_spectrograms = []
        for target_file in target_files:
            print(f"Loading target file: {target_file}")
            target_waveform, _ = torchaudio.load(target_file)
            print(f"Loaded target file with shape: {target_waveform.shape}")
            target_spectrogram = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(target_waveform)
            target_spectrograms.append(target_spectrogram)

        target_spectrograms = torch.stack(target_spectrograms, dim=0)
        target_chunks = self.chunk_spectrogram(target_spectrograms, is_input=False)
        print(f"Target spectrograms shape: {target_spectrograms.shape}, Target chunks shape: {target_chunks.shape}")

        return input_chunks, target_chunks

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
        
        return chunks

def load_data(dataset_path, batch_size, chunk_size, hop_length, n_fft):
    train_dataset = CustomDataset(dataset_path, chunk_size, hop_length, n_fft)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
    def __init__(self, input_size, hidden_size=512, degree=3):
        super(KANModel, self).__init__()
        self.input_layer = KANLayer(input_size, hidden_size, degree)
        self.hidden_layer1 = KANLayer(hidden_size, hidden_size, degree)
        self.hidden_layer2 = KANLayer(hidden_size, hidden_size, degree)
        self.output_layer = KANLayer(hidden_size, 7 * hidden_size, degree)  # Output 7 stems with hidden_size each
        print("KANModel initialization completed.")

    def forward(self, x):
        try:
            print(f"Input shape: {x.shape}")
            # Ensure the input tensor has the correct shape for the linear layers
            batch_size, num_channels, num_frequencies, num_timesteps, chunk_size = x.shape
            x = x.view(batch_size, num_channels * num_frequencies * num_timesteps * chunk_size)

            input_size = self.input_layer.linear.in_features
            if x.shape[1] != input_size:
                print(f"Input tensor shape mismatch: Expected {input_size} features, but got {x.shape[1]}")
                raise ValueError("Input tensor shape mismatch")

            print("Passing input through the input layer")
            x = self.input_layer(x)
            print(f"Input layer output shape: {x.shape}")
            
            print("Passing input through hidden layer 1")
            x = self.hidden_layer1(x)
            print(f"Hidden layer 1 output shape: {x.shape}")
            
            print("Passing input through hidden layer 2")
            x = self.hidden_layer2(x)
            print(f"Hidden layer 2 output shape: {x.shape}")
            
            print("Passing input through the output layer")
            x = self.output_layer(x)
            print(f"Output layer output shape: {x.shape}")
            
            # Reshape the output tensor to match the target shape
            x = x.view(batch_size, 7, num_channels, num_frequencies, num_timesteps, chunk_size)
            return x
        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise e

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}')
            # Clear memory after each batch
            del inputs, targets, outputs
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during training batch {batch_idx+1}: {e}")
            raise e
    return train_loss / len(train_loader)

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def main():
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

    train_loader = load_data(args.dataset, args.batch_size, args.chunk_size, args.hop_length, args.n_fft)
    
    # Initialize model, loss function, and optimizer
    input_size = args.chunk_size * (args.n_fft // 2 + 1) * 483 * 1  # Adjusted input size calculation
    model = KANModel(input_size=input_size, hidden_size=args.hidden_size).to(device)
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
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            raise e

if __name__ == "__main__":
    main()
