# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems=7):
        super(KANWithDepthwiseConv, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.fc = nn.Conv2d(out_channels, num_stems * in_channels, kernel_size=1)
        self.num_stems = num_stems
        self.in_channels = in_channels

    def forward(self, x):
        x = checkpoint.checkpoint(self._forward_impl, x, use_reentrant=True)
        return x

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_stems, self.in_channels, x.size(2), x.size(3))
        return x

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems=7):
    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems=num_stems)
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model
