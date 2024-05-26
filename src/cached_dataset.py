import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset  # Add this import statement
import tempfile
import numpy as np

def random_padding(waveform, target_length, n_fft):
    current_length = waveform.size(1)
    if current_length >= target_length:
        return waveform[:, :target_length]

    # Calculate padding to ensure the waveform length is divisible by hop_length
    hop_length = n_fft // 2
    padding_needed = (hop_length - (current_length % hop_length)) % hop_length
    total_padding = padding_needed

    # Ensure that total_padding is at least zero and not more than half n_fft
    total_padding = min(max(0, total_padding), n_fft // 2 - 1)

    # Padding added only to the right (end) of the waveform
    left_padding = 0
    right_padding = total_padding

    padding = (left_padding, right_padding)
    waveform = F.pad(waveform, padding, "constant", 0)
    return waveform

class StemSeparationDataset(Dataset):
    def __init__(self, data_dir, n_mels, target_length, n_fft, num_stems, cache_dir):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.target_length = target_length
        self.n_fft = n_fft
        self.num_stems = num_stems
        self.cache_dir = cache_dir

        self.input_files = []
        self.stem_files = [[] for _ in range(num_stems)]

        stem_mapping = {
            'bass': 0,
            'drums': 1,
            'guitar': 2,
            'keys': 3,
            'noise': 4,
            'other': 5,
            'vocals': 6
        }

        for file_name in os.listdir(data_dir):
            if file_name.startswith('input_'):
                self.input_files.append(file_name)
            elif file_name.startswith('target_'):
                stem_name = file_name.split('_')[1]
                if stem_name in stem_mapping:
                    stem_idx = stem_mapping[stem_name]
                    self.stem_files[stem_idx].append(file_name)

        # Ensure there is at least one file in each stem category
        for stem_list in self.stem_files:
            if len(stem_list) == 0:
                raise ValueError("Each stem category must contain at least one file.")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        input_path = os.path.join(self.data_dir, input_file)
        input_waveform, sr = torchaudio.load(input_path)

        # Preprocess input_waveform
        input_waveform = torchaudio.transforms.Resample(sr, 44100)(input_waveform)
        input_waveform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft)(input_waveform)

        target_spectrograms = []

        for stem in range(self.num_stems):
            if idx >= len(self.stem_files[stem]):
                stem_spectrogram = np.zeros_like(input_waveform.numpy())
            else:
                stem_file = self.stem_files[stem][idx]
                stem_path = os.path.join(self.data_dir, stem_file)
                stem_waveform, sr = torchaudio.load(stem_path)

                # Preprocess stem_waveform
                stem_waveform = torchaudio.transforms.Resample(sr, 44100)(stem_waveform)
                stem_waveform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft)(stem_waveform)
                stem_spectrogram = stem_waveform.numpy()

            target_spectrograms.append(torch.tensor(stem_spectrogram, dtype=torch.float32))

        input_spectrogram = torch.tensor(input_waveform.numpy(), dtype=torch.float32)
        return input_spectrogram, target_spectrograms
