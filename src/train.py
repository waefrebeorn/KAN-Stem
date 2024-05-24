import os
import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Define the CustomDataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, target_length=256):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.target_length = target_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        spectrogram = self._waveform_to_spectrogram(waveform)
        return spectrogram

    def _waveform_to_spectrogram(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=128)(waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()  # Normalize spectrogram
        if spectrogram.size(-1) > self.target_length:
            spectrogram = spectrogram[:, :, :self.target_length]
        else:
            padding = self.target_length - spectrogram.size(-1)
            spectrogram = nn.functional.pad(spectrogram, (0, padding))
        return spectrogram

# Define the KANLinear class
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))

# Define the KAN class
class KAN(nn.Module):
    def __init__(self, in_features):
        super(KAN, self).__init__()
        self.layer1 = KANLinear(in_features, 512)
        self.layer2 = KANLinear(512, 256)
        self.output_layer = nn.Linear(256, in_features)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the spectrogram
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        x = x.reshape(x.size(0), 128, -1)  # Reshape to spectrogram dimensions
        return x

def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, writer, checkpoint_dir, save_interval, in_features):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = data.to(device)
        targets = data.to(device)  # Ensure targets have the same shape as inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0
    # Save checkpoint at specified intervals
    if (epoch + 1) % save_interval == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'in_features': in_features
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

def main():
    parser = argparse.ArgumentParser(description='KAN Training Script')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval to save checkpoints (in epochs)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = CustomDataset(args.data_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True if args.use_cuda else False)

    # Calculate in_features based on n_mels and target_length
    sample_spectrogram = dataset[0]
    in_features = sample_spectrogram.size(0) * sample_spectrogram.size(2)  # 128 * 256 = 32768

    model = KAN(in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, args.num_epochs, writer, args.checkpoint_dir, args.save_interval, in_features)

    writer.close()
    logger.info('Training Finished')

if __name__ == '__main__':
    main()
