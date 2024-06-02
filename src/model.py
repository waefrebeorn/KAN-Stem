import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import numpy as np

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
        conv1_cache_path = self.cache_activation(x, 'conv1')

        x = F.relu(self.conv2(x.clone()))  # Clone x before passing to conv2
        x = self.pool2(x)
        conv2_cache_path = self.cache_activation(x, 'conv2')

        x = F.relu(self.conv3(x.clone()))  # Clone x before passing to conv3
        x = self.pool3(x)
        conv3_cache_path = self.cache_activation(x, 'conv3')

        x = F.relu(self.conv4(x.clone()))  # Clone x before passing to conv4
        x = self.pool4(x)
        x = self.pool5(x)

        x = self.flatten(x)

        # Initialize fc1 and fc2 based on the actual input size only once
        if self.fc1 is None or self.fc2 is None:
            conv_output_size = self._get_conv_output([self.n_mels, self.target_length])
            self._initialize_fc_layers(conv_output_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_stems, self.n_mels, self.target_length)

        # Clean up cached activation files
        for cache_path in [conv1_cache_path, conv2_cache_path, conv3_cache_path]:
            if os.path.exists(cache_path):
                os.remove(cache_path)

        return x

    def cache_activation(self, x, name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_metadata_path = os.path.join(self.cache_dir, "cache_metadata.pt")

        # Load existing cache metadata
        if os.path.exists(cache_metadata_path):
            cache_metadata = torch.load(cache_metadata_path)
        else:
            cache_metadata = {}

        # Save the activation tensor to a temporary file
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as temp_file:
            x = x.detach().cpu().numpy()  # Move tensor to CPU before saving
            np.save(temp_file.name, x)

            # Update cache metadata with the path of the saved tensor
            cache_key = f"{name}_{temp_file.name}"
            cache_metadata[cache_key] = temp_file.name

            # Save the updated cache metadata
            torch.save(cache_metadata, cache_metadata_path)
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

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device, freeze_fc_layers=False):
    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, None, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.train()  # Switch to training mode

    if freeze_fc_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

    model.to(device)
    return model
