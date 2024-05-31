import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import numpy as np
import os

class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems, cache_dir, device):
        super(KANWithDepthwiseConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(3, 3), padding=1).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=(3, 3), padding=1).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.conv4 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=(3, 3), padding=1).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.flatten = nn.Flatten().to(device)

        self.n_mels = n_mels
        self.target_length = target_length
        self.num_stems = num_stems
        self.cache_dir = cache_dir
        self.device = device

        self.fc1 = None  # To be initialized later
        self.fc2 = None  # To be initialized later
    
    def _initialize_fc_layers(self, conv_output_size):
        self.fc1 = nn.Linear(conv_output_size, 1024).to(self.device)
        self.fc2 = nn.Linear(1024, self.n_mels * self.target_length * self.num_stems).to(self.device)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.rand(1, 1, shape[0], shape[1]).to(self.device)
            x = self.pool1(F.relu(self.conv1(x.clone())))
            x = self.pool2(F.relu(self.conv2(x.clone())))
            x = self.pool3(F.relu(self.conv3(x.clone())))
            x = self.pool4(F.relu(self.conv4(x.clone())))
            x = self.pool5(x)
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, x):
        x = x.to(self.device)  # Ensure x is on the correct device
        x = F.relu(self.conv1(x.clone()))  # Clone x before passing to conv1
        x = self.pool1(x)
        print(f'After conv1 and pool1: {x.shape}')  # Debug print
        conv1_cache_path = self.cache_activation(x, 'conv1')

        x = F.relu(self.conv2(x.clone()))  # Clone x before passing to conv2
        x = self.pool2(x)
        print(f'After conv2 and pool2: {x.shape}')  # Debug print
        conv2_cache_path = self.cache_activation(x, 'conv2')

        x = F.relu(self.conv3(x.clone()))  # Clone x before passing to conv3
        x = self.pool3(x)
        print(f'After conv3 and pool3: {x.shape}')  # Debug print
        conv3_cache_path = self.cache_activation(x, 'conv3')

        x = F.relu(self.conv4(x.clone()))  # Clone x before passing to conv4
        x = self.pool4(x)
        print(f'After conv4 and pool4: {x.shape}')  # Debug print
        x = self.pool5(x)
        print(f'After additional pool5: {x.shape}')  # Debug print

        x = self.flatten(x)
        print(f'After flatten: {x.shape}')  # Debug print

        # Initialize fc1 and fc2 based on the actual input size only once
        if self.fc1 is None or self.fc2 is None:
            conv_output_size = self._get_conv_output([self.n_mels, self.target_length])
            self._initialize_fc_layers(conv_output_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_stems, self.n_mels, self.target_length)
        print(f'After fc1 and fc2: {x.shape}')  # Debug print

        # Clean up cached activation files
        for cache_path in [conv1_cache_path, conv2_cache_path, conv3_cache_path]:
            if os.path.exists(cache_path):
                os.remove(cache_path)

        return x

    def cache_activation(self, x, name):
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as temp_file:
            np.save(temp_file.name, x.detach().cpu().numpy())
            return temp_file.name

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, n_mels=128, target_length=256, device="cpu"):
        super(KANDiscriminator, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(128).to(device)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1).to(device)
        self.bn3 = nn.BatchNorm2d(out_channels).to(device)

        # Calculate the flattened size based on the convolutional layers' output
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, n_mels, target_length, device=device)
            dummy_output = self._forward_conv_layers(dummy_input)
            flattened_size = dummy_output.view(-1).shape[0]
        
        # Modify the input size of fc1 to match the calculated flattened size
        self.fc1 = nn.Linear(flattened_size, 1).to(device)

    def _forward_conv_layers(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.sigmoid(self.fc1(x))
        return x
        
def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device):
    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, None, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # Load with strict=False to ignore unexpected keys
    model.to(device)
    model.eval()
    return model
