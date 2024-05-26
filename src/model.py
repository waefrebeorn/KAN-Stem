import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import tempfile
import numpy as np
import os

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems, cache_dir):
        super(KANWithDepthwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # Additional pooling layer
        self.flatten = nn.Flatten()
        
        self.n_mels = n_mels
        self.target_length = target_length
        self.num_stems = num_stems
        self.cache_dir = cache_dir
        self._initialize_fc1(in_channels, n_mels, target_length)

    def _initialize_fc1(self, in_channels, n_mels, target_length):
        conv_output_size = self._get_conv_output((in_channels, n_mels, target_length))
        print(f'Conv output size: {conv_output_size}')  # Debug print
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, n_mels * target_length * num_stems)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.rand(1, *shape)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(x)  # Additional pooling layer
            x = self.flatten(x)
            return x.shape[1]

    def cache_activation(self, x, name):
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as temp_file:
            np.save(temp_file.name, x.detach().cpu().numpy())
            return temp_file.name

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        print(f'After conv1 and pool1: {x.shape}')  # Debug print
        conv1_cache_path = self.cache_activation(x, 'conv1')

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        print(f'After conv2 and pool2: {x.shape}')  # Debug print
        conv2_cache_path = self.cache_activation(x, 'conv2')

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        print(f'After conv3 and pool3: {x.shape}')  # Debug print
        x = self.pool4(x)  # Additional pooling layer
        print(f'After additional pool4: {x.shape}')  # Debug print
        conv3_cache_path = self.cache_activation(x, 'conv3')

        x = self.flatten(x)
        print(f'After flatten: {x.shape}')  # Debug print
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_stems, self.n_mels, self.target_length)
        print(f'After fc1 and fc2: {x.shape}')  # Debug print

        # Clean up cached activation files
        for cache_path in [conv1_cache_path, conv2_cache_path, conv3_cache_path]:
            if os.path.exists(cache_path):
                os.remove(cache_path)

        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length):
        super(KANDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_mels = n_mels
        self.target_length = target_length

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.fc1 = nn.Linear(out_channels * n_mels * target_length, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc1(x))
        return x

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device):
    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
