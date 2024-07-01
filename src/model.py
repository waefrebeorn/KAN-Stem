import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import h5py
import math
import gc
from typing import Dict

logger = logging.getLogger(__name__)

def purge_vram():
    torch.cuda.empty_cache()
    gc.collect()

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -1.5, grid_max: float = 1.5, num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids, device='cuda')
        self.register_buffer("grid", grid)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class BSRBF_KANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_size=5, spline_order=3, base_activation=torch.nn.ReLU, grid_range=[-1.5, 1.5]):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim).cuda()
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.base_activation = base_activation().cuda()
        self.input_dim = input_dim

        self.base_weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim, device='cuda'))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim * (grid_size + spline_order), device='cuda'))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size + spline_order).cuda()

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1, device='cuda') * h + grid_range[0]).expand(self.input_dim, -1).contiguous()
        self.register_buffer("grid", grid)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(2) == self.input_dim
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, :, 1:])
        assert bases.size() == (x.size(0), x.size(1), self.input_dim, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        bs_output = self.b_splines(x).view(x.size(0), x.size(1), -1)
        rbf_output = self.rbf(x).view(x.size(0), x.size(1), -1)
        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        return base_output + bsrbf_output

class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation_rate, grid_size=5, spline_order=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, groups=channels, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(channels).cuda()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(channels).cuda()
        self.dropout = nn.Dropout(p=0.1).cuda()

        self.bsrbf_layer = BSRBF_KANLayer(channels, channels, grid_size, spline_order).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        x += self.bsrbf_layer(residual.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(residual)
        return F.relu(x)

class ContextAggregationNetwork(nn.Module):
    def __init__(self, channels):
        super(ContextAggregationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(channels).cuda()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(channels).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MemoryEfficientStemSeparationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_mels=64, target_length=22050):
        super(MemoryEfficientStemSeparationModel, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            self.conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            self.conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            self.conv_block(128, 256, kernel_size=3, stride=2, padding=1)
        ])

        # Decoder 
        self.decoder = nn.ModuleList([
            self.conv_block(256, 128, kernel_size=3, stride=1, padding=1),
            self.conv_block(128, 64, kernel_size=3, stride=1, padding=1),
            self.conv_block(64, 32, kernel_size=3, stride=1, padding=1)
        ])
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.encoder:
            x = checkpoint(encoder_layer, x, use_reentrant=False)
        return x

    def _forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        for decoder_layer in self.decoder:
            x = checkpoint(decoder_layer, x, use_reentrant=False)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 3:
            raise ValueError(f"Expected input with 3 channels but got {x.size(1)} channels")

        x = self._forward_encoder(x)
        x = F.interpolate(x, size=(self.n_mels, self.target_length), mode='bilinear', align_corners=False)
        x = self._forward_decoder(x)
        x = self.final_conv(x)

        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, n_mels=127, target_length=44036, device="cuda", channel_multiplier=1.0):
        super(KANDiscriminator, self).__init__()

        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels, int(out_channels * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(int(out_channels * channel_multiplier)).cuda()
        self.conv2 = nn.Conv2d(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(int(out_channels * 2 * channel_multiplier)).cuda()
        self.conv3 = nn.Conv2d(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn3 = nn.InstanceNorm2d(int(out_channels * 4 * channel_multiplier)).cuda()

        self.fc1 = None

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed
        if x.dim() > 4:
            raise ValueError(f"Invalid input shape. Expected 3 or 4 dimensions but got {x.dim()}")

        x = x.to(self.device)  # Ensure the data is on the correct device
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.fc1 is None or self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 1).to(self.device)
            nn.init.xavier_normal_(self.fc1.weight)

        x = torch.sigmoid(self.fc1(x))
        return x

def load_from_cache(cache_file_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:], device=device).float()
            target_data = torch.tensor(f['target'][:], device=device).float()
            zero_durations = torch.tensor(f['zero_durations'][:], device=device).float()
            input_data = input_data.unsqueeze(0) if input_data.dim() == 3 else input_data
            target_data = target_data.unsqueeze(0) if target_data.dim() == 3 else target_data
        return {'input': input_data, 'target': target_data, 'zero_durations': zero_durations}
    except Exception as e:
        logger.error(f"Error loading from cache file '{cache_file_path}': {e}")
        raise

def reassemble_with_zero_gaps(data: torch.Tensor, zero_durations: torch.Tensor, segment_length: int = 87) -> torch.Tensor:
    """
    Reassemble the audio tensor with zero gaps as per the zero_durations information.
    """
    reassembled = []
    current_index = 0
    for duration in zero_durations:
        if duration > 0:
            zero_tensor = torch.zeros((1, 3, 32, segment_length * duration), device=data.device)
            reassembled.append(zero_tensor)
        reassembled.append(data[:, :, :, current_index:current_index + segment_length])
        current_index += segment_length
    return torch.cat(reassembled, dim=-1)

def load_model(checkpoint_path: str, in_channels: int, out_channels: int, n_mels: int, target_length: int, device: str = "cuda") -> nn.Module:
    model = MemoryEfficientStemSeparationModel(in_channels, out_channels, n_mels, target_length).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    logger.info("Model script executed directly.")
else:
    logger.info("Model script imported as a module.")
