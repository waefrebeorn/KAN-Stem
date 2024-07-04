import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import h5py
import gc

logger = logging.getLogger(__name__)

def purge_vram():
    torch.cuda.empty_cache()
    gc.collect()

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

class MemoryEfficientStemSeparationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_mels=32, target_length=22050):
        super(MemoryEfficientStemSeparationModel, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length
        self.scaler = torch.cuda.amp.GradScaler()

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
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1).cuda()

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

    def _forward_encoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str) -> torch.Tensor:
        log_memory_usage("Before Encoder")
        logger.info(f"Encoder input shape: {x.shape}")
        cache_file_name = f'{cache_prefix}_encoder_{identifier}.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
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

    def _forward_combiner(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str) -> torch.Tensor:
        log_memory_usage("Before Combiner")
        logger.info(f"Combiner input shape: {x.shape}")
        cache_file_name = f'{cache_prefix}_combiner_{identifier}.h5'
        cache_file_path = os.path.join(model_cache_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            x = load_from_cache(model_cache_dir, cache_file_name, x.device)
        else:
            x = checkpoint(self.combiner, x, use_reentrant=False)
            save_to_cache(model_cache_dir, cache_file_name, x)
        logger.info(f"Combiner output shape: {x.shape}")
        purge_vram()
        log_memory_usage("After Combiner")
        return x

    def _forward_decoder(self, x: torch.Tensor, model_cache_dir: str, cache_prefix: str, identifier: str) -> torch.Tensor:
        log_memory_usage("Before Decoder")
        logger.info(f"Decoder input shape: {x.shape}")
        for idx, upsampler_layer in enumerate(self.upsampler):
            cache_file_name = f'{cache_prefix}_decoder_{idx}_{identifier}.h5'
            cache_file_path = os.path.join(model_cache_dir, cache_file_name)
            if os.path.exists(cache_file_path):
                x = load_from_cache(model_cache_dir, cache_file_name, x.device)
            else:
                with torch.cuda.amp.autocast():
                    x = checkpoint(upsampler_layer, x, use_reentrant=False)
                save_to_cache(model_cache_dir, cache_file_name, x)
                logger.info(f"Decoder layer {idx} output shape: {x.shape}")

        if x.shape[-1] > self.target_length:
            x = x[..., :self.target_length]
            logger.info(f"Adjusted decoder layer output shape: {x.shape}")

        purge_vram()
        log_memory_usage("After Decoder")
        return x

    def forward(self, x, model_cache_dir: str, cache_prefix: str, identifier: str):
        log_memory_usage("Start Forward")
        original_segments = x.shape[1] if x.dim() == 6 else 1 
        if x.dim() == 5:
            x = self._reshape_input_tensor(x)
        with torch.cuda.amp.autocast():
            x = self._forward_encoder(x, model_cache_dir, cache_prefix, identifier)
            x = self._forward_combiner(x, model_cache_dir, cache_prefix, identifier)
            x = self._forward_decoder(x, model_cache_dir, cache_prefix, identifier)
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
        if x.dim() == 6:  # Adjust this part to handle 6D input properly
            x = x.squeeze(2)
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
