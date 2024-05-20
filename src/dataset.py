import os
import torch
from torch.utils.data import Dataset
import torchaudio

class MyAudioDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.audio_files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.dataset_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, waveform  # Assuming the input and target are the same for now
