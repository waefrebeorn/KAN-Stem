import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import h5py
import math
import gc

logger = logging.getLogger(__name__)

def purge_vram():
    torch.cuda.empty_cache()
    gc.collect()

def save_to_cache(cache_dir: str, cache_file_name: str, tensor: torch.Tensor):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    logger.info(f"Saving tensor to cache file: {cache_file_path}")
    with h5py.File(cache_file_path, 'w') as f:
        f.create_dataset('tensor', data=tensor.detach().cpu().numpy())
    logger.info(f"Saved tensor shape: {tensor.shape}")

def load_from_cache(cache_dir: str, cache_file_name: str, device: torch.device) -> torch.Tensor:
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    logger.info(f"Loading tensor from cache file: {cache_file_path}")
    with h5py.File(cache_file_path, 'r') as f:
        if 'tensor' not in f:
            raise KeyError(f"Tensor not found in cache file: {cache_file_path}")
        tensor = torch.tensor(f['tensor'][:], device=device).float()
    logger.info(f"Loaded tensor shape: {tensor.shape}")
    return tensor

def clear_cache(cache_dir: str):
    for file_name in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting cache file {file_path}: {e}")

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -1.5, grid_max: float = 1.5, num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids, device='cuda')
        self.register_buffer("grid", grid)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        cache_file_name = f'{cache_prefix}_rbf.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            return load_from_cache(model_cache_dir, cache_file_name, x.device)
        
        output = torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
        save_to_cache(model_cache_dir, cache_file_name, output)
        purge_vram()
        return output

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

    def b_splines(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        assert x.dim() == 3 and x.size(2) == self.input_dim
        cache_file_name = f'{cache_prefix}_bsplines.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            return load_from_cache(model_cache_dir, cache_file_name, x.device)
        
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, :, 1:])
        assert bases.size() == (x.size(0), x.size(1), self.input_dim, self.grid_size + self.spline_order)
        save_to_cache(model_cache_dir, cache_file_name, bases)
        purge_vram()
        return bases.contiguous()

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        x = self.layernorm(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        bs_output = self.b_splines(x, model_cache_dir, cache_prefix).view(x.size(0), x.size(1), -1)
        rbf_output = self.rbf(x, model_cache_dir, cache_prefix).view(x.size(0), x.size(1), -1)
        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        purge_vram()
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

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        cache_file_name = f'{cache_prefix}_attention.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            return load_from_cache(model_cache_dir, cache_file_name, x.device)

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        output = x * y

        save_to_cache(model_cache_dir, cache_file_name, output)
        purge_vram()
        return output

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation_rate, grid_size=5, spline_order=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(channels).cuda()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(channels).cuda()
        self.dropout = nn.Dropout(p=0.1).cuda()

        self.bsrbf_layer = BSRBF_KANLayer(channels, channels, grid_size, spline_order).cuda()

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, chunk_size: int = 10) -> torch.Tensor:
        residual = x
        batch_size = x.size(0)
        results = []

        for i in range(0, batch_size, chunk_size):
            x_chunk = x[i:i + chunk_size]
            cache_file_name = f'{cache_prefix}_residual_{i}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)

            if os.path.exists(cache_file_path):
                x_chunk = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                x_chunk = F.relu(self.bn1(self.conv1(x_chunk)))
                x_chunk = self.bn2(self.conv2(x_chunk))
                x_chunk = self.dropout(x_chunk)
                x_chunk += self.bsrbf_layer(residual[i:i + chunk_size].flatten(2).transpose(1, 2), model_cache_dir, f'{cache_prefix}_bsrbf_{i}').transpose(1, 2).view_as(x_chunk)
                x_chunk = F.relu(x_chunk)
                save_to_cache(model_cache_dir, cache_file_name, x_chunk)

            results.append(x_chunk)

        purge_vram()
        return torch.cat(results, dim=0)

class ContextAggregationNetwork(nn.Module):
    def __init__(self, channels):
        super(ContextAggregationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(channels).cuda()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(channels).cuda()

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        cache_file_name = f'{cache_prefix}_context.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            return load_from_cache(model_cache_dir, cache_file_name, x.device)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        save_to_cache(model_cache_dir, cache_file_name, x)
        purge_vram()
        return x

class MemoryEfficientStemSeparationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_mels=32, target_length=22050):
        super(MemoryEfficientStemSeparationModel, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length

        self.channel_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ).cuda() for _ in range(in_channels)
        ])

        self.combiner = nn.Sequential(
            nn.Conv2d(256 * in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ).cuda()

        self.upsampler = nn.ModuleList([
            self.deconv_block(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.deconv_block(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.deconv_block(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1).cuda()

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ).cuda()

    @staticmethod
    def deconv_block(in_channels, out_channels, kernel_size, stride, padding, output_padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ).cuda()

    def _forward_encoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        processed_segments = []
        for idx, segment in enumerate(x):
            cache_file_name = f'{cache_prefix}_encoder_{idx}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)
            if os.path.exists(cache_file_path):
                processed_segment = load_from_cache(model_cache_dir, cache_file_name, segment.device)
            else:
                segment = segment.squeeze(1)
                processed_channels = [checkpoint(channel_processor, segment[i:i+1, :, :].unsqueeze(0), use_reentrant=False) for i, channel_processor in enumerate(self.channel_processors)]
                processed_segment = torch.cat(processed_channels, dim=1)
                save_to_cache(model_cache_dir, cache_file_name, processed_segment)
            processed_segments.append(processed_segment)
        x = torch.cat(processed_segments, dim=0)
        purge_vram()
        return x

    def _forward_combiner(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        cache_file_name = f'{cache_prefix}_combiner.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            x = load_from_cache(model_cache_dir, cache_file_name, x.device)
        else:
            x = checkpoint(self.combiner, x, use_reentrant=False)
            save_to_cache(model_cache_dir, cache_file_name, x)
        purge_vram()
        return x

    def _forward_decoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str) -> torch.Tensor:
        for idx, upsampler_layer in enumerate(self.upsampler):
            cache_file_name = f'{cache_prefix}_decoder_{idx}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)
            if os.path.exists(cache_file_path):
                x = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                x = checkpoint(upsampler_layer, x, use_reentrant=False)
                save_to_cache(model_cache_dir, cache_file_name, x)
        purge_vram()
        return x

    def forward(self, x, model_cache_dir: str, cache_prefix: str):
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self._forward_encoder(x, model_cache_dir, cache_prefix)
        x = self._forward_combiner(x, model_cache_dir, cache_prefix)
        x = self._forward_decoder(x, model_cache_dir, cache_prefix)
        x = self.final_conv(x)
        # Ensure output matches the target length
        x = x[:, :, :, :self.target_length]
        purge_vram()
        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, n_mels=32, target_length=22050, device="cuda", channel_multiplier=1.0):
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

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, chunk_size: int = 10) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() > 4:
            raise ValueError(f"Invalid input shape. Expected 3 or 4 dimensions but got {x.dim()}")

        x = x.to(self.device)
        
        batch_size = x.size(0)
        results = []
        
        for i in range(0, batch_size, chunk_size):
            x_chunk = x[i:i + chunk_size]
            cache_file_name = f'{cache_prefix}_discriminator_{i}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)

            if os.path.exists(cache_file_path):
                x_chunk = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                x_chunk = self._forward_conv_layers(x_chunk)
                x_chunk = x_chunk.view(x_chunk.size(0), -1)
                
                if self.fc1 is None or self.fc1.in_features != x_chunk.shape[1]:
                    self.fc1 = nn.Linear(x_chunk.shape[1], 1).to(self.device)
                    nn.init.xavier_normal_(self.fc1.weight)
                
                x_chunk = torch.sigmoid(self.fc1(x_chunk))
                save_to_cache(model_cache_dir, cache_file_name, x_chunk)
            
            results.append(x_chunk)
        
        purge_vram()
        return torch.cat(results, dim=0)

def load_model(checkpoint_path: str, in_channels: int, out_channels: int, n_mels: int, target_length: int, device: str = "cuda") -> nn.Module:
    model = MemoryEfficientStemSeparationModel(in_channels, out_channels, n_mels, target_length).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    logger.info("Model script executed directly.")
else:
    logger.info("Model script imported as a module.")
