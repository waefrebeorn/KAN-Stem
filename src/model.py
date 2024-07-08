import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import logging
import h5py
import gc
import math
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def purge_vram():
    try:
        torch.cuda.empty_cache()
        logger.info("Successfully purged GPU cache.")
    except Exception as e:
        logger.error(f"Error purging GPU cache: {e}", exc_info=True)
    try:
        gc.collect()
        logger.info("Successfully performed garbage collection.")
    except Exception as e:
        logger.error(f"Error during garbage collection: {e}", exc_info=True)

def log_memory_usage(tag):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    logger.info(f"[{tag}] Allocated memory: {allocated / (1024 ** 3):.2f} GB, Reserved memory: {reserved / (1024 ** 3):.2f} GB")

def save_to_cache(cache_dir: str, cache_file_name: str, tensor: torch.Tensor):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    logger.info(f"Saving tensor to cache file: {cache_file_path}")
    with h5py.File(cache_file_path, 'w') as f:
        f.create_dataset('tensor', data=tensor.detach().cpu().numpy())
    logger.info(f"Saved tensor shape: {tensor.shape}")
    purge_vram()

def load_from_cache(cache_dir: str, cache_file_name: str, device: torch.device) -> torch.Tensor:
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    logger.info(f"Loading tensor from cache file: {cache_file_path}")
    with h5py.File(cache_file_path, 'r') as f:
        if 'tensor' not in f:
            raise KeyError(f"Tensor not found in cache file: {cache_file_path}")
        tensor = torch.tensor(f['tensor'][:], device=device).float()
    logger.info(f"Loaded tensor shape: {tensor.shape}")
    purge_vram()
    return tensor

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -1.5, grid_max: float = 1.5, num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids, device='cuda')
        self.register_buffer("grid", grid)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            output = torch.exp(-((x[..., None] - x.grid) / x.denominator) ** 2)
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

    @staticmethod
    def b_splines(x: torch.Tensor, grid: torch.Tensor, spline_order: int, grid_size: int) -> torch.Tensor:
        assert x.dim() == 3 and x.size(2) == grid.size(0)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, :, 1:])
        assert bases.size() == (x.size(0), x.size(1), grid.size(0), grid_size + spline_order)
        purge_vram()
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        with torch.cuda.amp.autocast():
            base_output = F.linear(self.base_activation(x), self.base_weight)

            bs_output = self.b_splines(x, self.grid, self.spline_order, self.grid_size).view(x.size(0), x.size(1), -1)
            rbf_output = self.rbf(x).view(x.size(0), x.size(1), -1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        with torch.cuda.amp.autocast():
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            output = x * y

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

    def forward(self, x: torch.Tensor, chunk_size: int = 10) -> torch.Tensor:
        residual = x
        batch_size = x.size(0)
        results = []

        for i in range(0, batch_size, chunk_size):
            x_chunk = x[i:i + chunk_size]
            with torch.cuda.amp.autocast():
                x_chunk = F.relu(self.bn1(self.conv1(x_chunk)))
                x_chunk = self.bn2(self.conv2(x_chunk))
                x_chunk = self.dropout(x_chunk)
                x_chunk += self.bsrbf_layer(residual[i:i + chunk_size].flatten(2).transpose(1, 2)).transpose(1, 2).view_as(x_chunk)
                x_chunk = F.relu(x_chunk)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))

            purge_vram()
            return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            return self.relu(x)

class MemoryEfficientStemSeparationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_mels=32, target_length=22050):
        super(MemoryEfficientStemSeparationModel, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length
        self.scaler = torch.cuda.amp.GradScaler()

        self.channel_processors = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(1, 32),
                DepthwiseSeparableConv(32, 64, stride=2),
                DepthwiseSeparableConv(64, 128, stride=2),
                DepthwiseSeparableConv(128, 256, stride=2)
            ).cuda() for _ in range(in_channels)
        ])

        self.combiner = nn.Sequential(
            DepthwiseSeparableConv(256 * in_channels, 256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ).cuda()

        self.upsampler = nn.ModuleList([
            self.deconv_block(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.deconv_block(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.deconv_block(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1).cuda()

        self.attention_layer = AttentionLayer(256)
        self.residual_block = ResidualBlock(256, dilation_rate=2)
        self.context_aggregation_network = ContextAggregationNetwork(256)

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

    def _reshape_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, segments, _, in_channels, n_mels, length = x.shape
        x = x.view(batch_size * segments, in_channels, n_mels, length)
        return x

    def _reshape_output_tensor(self, x: torch.Tensor, original_segments: int) -> torch.Tensor:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() == 4:
            batch_size, in_channels, n_mels, length = x.shape
            x = x.view(1, original_segments, 1, in_channels, n_mels, length)
        elif x.dim() == 5:
            batch_size, segments, in_channels, n_mels, length = x.shape
            x = x.view(batch_size // original_segments, original_segments, 1, in_channels, n_mels, length)
        return x

    def _forward_encoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool) -> torch.Tensor:
        log_memory_usage("Before Encoder")
        logger.info(f"Encoder input shape: {x.shape}")
        
        cache_file_name = f'{cache_prefix}_encoder_{identifier}.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)

        if os.path.exists(cache_file_path) and not update_cache:
            x = load_from_cache(model_cache_dir, cache_file_name, x.device)
        else:
            x = self._reshape_input_tensor(x)
            processed_segments = []
            for idx in range(x.size(0)):
                segment = x[idx:idx+1, :, :, :]
                processed_channels = [checkpoint(channel_processor, segment[:, i:i+1, :, :], use_reentrant=False) for i, channel_processor in enumerate(self.channel_processors)]
                processed_segment = torch.cat(processed_channels, dim=1)
                processed_segments.append(processed_segment)
            x = torch.cat(processed_segments, dim=0)
            
            save_to_cache(model_cache_dir, cache_file_name, x)
        
        logger.info(f"Encoder output shape: {x.shape}")
        purge_vram()
        log_memory_usage("After Encoder")
        return x

    def _forward_combiner(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool) -> torch.Tensor:
        log_memory_usage("Before Combiner")
        logger.info(f"Combiner input shape: {x.shape}")

        cache_file_name = f'{cache_prefix}_combiner_{identifier}.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)

        if os.path.exists(cache_file_path) and not update_cache:
            x = load_from_cache(model_cache_dir, cache_file_name, x.device)
        else:
            x = checkpoint(self.combiner, x, use_reentrant=False)
            x = self.attention_layer(x)
            x = self.context_aggregation_network(x)
            
            save_to_cache(model_cache_dir, cache_file_name, x)

        logger.info(f"Combiner output shape: {x.shape}")
        purge_vram()
        log_memory_usage("After Combiner")
        return x

    def _forward_decoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool) -> torch.Tensor:
        log_memory_usage("Before Decoder")
        logger.info(f"Decoder input shape: {x.shape}")
        for idx, upsampler_layer in enumerate(self.upsampler):
            cache_file_name = f'{cache_prefix}_decoder_{idx}_{identifier}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)
            if os.path.exists(cache_file_path) and not update_cache:
                x = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                with torch.cuda.amp.autocast():
                    x = checkpoint(upsampler_layer, x, use_reentrant=False)
                save_to_cache(model_cache_dir, cache_file_name, x)
                logger.info(f"Decoder layer {idx} output shape: {x.shape}")

            purge_vram()

        if x.shape[-1] > self.target_length:
            x = x[..., :self.target_length]
            logger.info(f"Adjusted decoder layer output shape: {x.shape}")

        purge_vram()
        log_memory_usage("After Decoder")
        return x

    def forward(self, x, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool = False) -> torch.Tensor:
        log_memory_usage("Start Forward")
        original_segments = x.shape[1] if x.dim() == 6 else 1 
        if x.dim() == 5:
            x = self._reshape_input_tensor(x)
        with torch.cuda.amp.autocast():
            x = self._forward_encoder(x, model_cache_dir, cache_prefix, identifier, update_cache)
            x = self._forward_combiner(x, model_cache_dir, cache_prefix, identifier, update_cache)
            x = self._forward_decoder(x, model_cache_dir, cache_prefix, identifier, update_cache)
            x = self.final_conv(x)
            logger.info(f"Final conv output shape: {x.shape}")

        if x.shape[-1] > self.target_length:
            x = x[..., :self.target_length]

        logger.info(f"Output shape before reshape: {x.shape}")
        x = self._reshape_output_tensor(x, original_segments)
        logger.info(f"Output shape after reshape: {x.shape}")

        log_memory_usage("End Forward")
        purge_vram()
        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, n_mels=32, target_length=22050, device="cuda", channel_multiplier=1.0):
        super(KANDiscriminator, self).__init__()

        self.device = device

        self.conv1 = DepthwiseSeparableConv(in_channels, int(out_channels * channel_multiplier)).cuda()
        self.conv2 = DepthwiseSeparableConv(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier)).cuda()
        self.conv3 = DepthwiseSeparableConv(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier)).cuda()

        self.fc1 = None

    def _reshape_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, segments, _, in_channels, n_mels, length = x.shape
        x = x.view(batch_size * segments, in_channels, n_mels, length)
        return x

    def _forward_conv_layers_with_cache(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool) -> torch.Tensor:
        results = []
        batch_size = x.size(0)
        chunk_size = 1  # Smaller chunk size to reduce memory usage

        logger.info(f"Batch size: {batch_size}, Chunk size: {chunk_size}")

        for i in range(0, batch_size, chunk_size):
            x_chunk = x[i:i + chunk_size]
            cache_file_name = f'{cache_prefix}_conv_{identifier}_{i}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        
            logger.info(f"Processing chunk {i // chunk_size + 1}/{batch_size // chunk_size}")
            if os.path.exists(cache_file_path) and not update_cache:
                logger.info(f"Loading from cache: {cache_file_name}")
                x_chunk = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                logger.info(f"Processing and caching chunk: {cache_file_name}")
                x_chunk = self._forward_conv_layers(x_chunk)
                save_to_cache(model_cache_dir, cache_file_name, x_chunk)
                purge_vram()

            results.append(x_chunk)

        return torch.cat(results, dim=0)

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        purge_vram()
        log_memory_usage("After conv1")
        x = self.conv2(x)
        purge_vram()
        log_memory_usage("After conv2")
        x = self.conv3(x)
        purge_vram()
        log_memory_usage("After conv3")
        return x

    def forward(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str, update_cache: bool = False) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 6:
            x = self._reshape_input_tensor(x)
        if x.dim() != 4:
            raise ValueError(f"Invalid input shape. Expected 4 dimensions but got {x.dim()}")

        x = x.to(self.device)

        batch_size = x.size(0)
        results = []

        chunk_size = 1  # Adjusting chunk size to 1 for memory efficiency
        for i in range(0, batch_size, chunk_size):
            x_chunk = x[i:i + chunk_size]
            x_chunk = self._forward_conv_layers_with_cache(x_chunk, model_cache_dir, cache_prefix, f'{identifier}_{i}', update_cache)
            x_chunk = x_chunk.view(x_chunk.size(0), -1)

            if self.fc1 is None or self.fc1.in_features != x_chunk.shape[1]:
                self.fc1 = nn.Linear(x_chunk.shape[1], 1).to(self.device)
                nn.init.xavier_normal_(self.fc1.weight)

            with torch.cuda.amp.autocast():
                x_chunk = torch.sigmoid(self.fc1(x_chunk))

            results.append(x_chunk)
            purge_vram()  # Purge VRAM after processing each chunk
    
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
