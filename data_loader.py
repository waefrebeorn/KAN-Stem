import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class MixedAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.wav_dir = os.path.join(root_dir, 'wav')
        self.file_names = [f for f in os.listdir(self.wav_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.wav_dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        # Downsample to reduce memory usage if needed
        if sample_rate > 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        return waveform, 16000  # Using a fixed sample rate for all audio

def get_data_loader(dataset_path, batch_size):
    dataset = MixedAudioDataset(root_dir=dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader
