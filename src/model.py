import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.utils.checkpoint as checkpoint
import h5py
import numpy as np
from cachetools import LRUCache, cached
from collections import defaultdict
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y.to(x.device)).view(b, c, 1, 1)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    @property
    def kernel_size(self):
        return self.depthwise.kernel_size

    @property
    def dilation(self):
        return self.depthwise.dilation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(ResidualBlock, self).__init__()
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.ins = nn.ModuleList([nn.InstanceNorm2d(out_channels) for _ in dilation_rates])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        for conv, ins in zip(self.convs, self.ins):
            x = checkpoint.checkpoint(conv, x, use_reentrant=False)
            x = F.relu(checkpoint.checkpoint(ins, x, use_reentrant=False))
        x = self.dropout(x)
        x += residual
        return F.relu(x)

class ContextAggregationNetwork(nn.Module):
    def __init__(self, channels):
        super(ContextAggregationNetwork, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.ins1 = nn.InstanceNorm2d(channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.ins2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        x = F.relu(checkpoint.checkpoint(self.ins1, self.conv1(x), use_reentrant=False))
        x = F.relu(checkpoint.checkpoint(self.ins2, self.conv2(x), use_reentrant=False))
        return x

class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems, cache_dir, device):
        super(KANWithDepthwiseConv, self).__init__()

        self.device = device
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, 'intermediate_cache.h5')
        self.index_file_path = os.path.join(cache_dir, 'cache_index.json')
        self.json_cache_file_path = os.path.join(cache_dir, 'intermediate_cache.json')

        self.cache_index = defaultdict(list)
        self.load_cache_index()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=(3, 3), padding=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.res_block1 = ResidualBlock(out_channels, out_channels).to(device)
        self.attention1 = AttentionLayer(out_channels).to(device)

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1)
        ).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.res_block2 = ResidualBlock(out_channels * 2, out_channels * 2).to(device)
        self.attention2 = AttentionLayer(out_channels * 2).to(device)

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(out_channels * 2, out_channels * 2, kernel_size=(3, 3), padding=1, dilation=2),
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=1)
        ).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)

        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(out_channels * 4, out_channels * 4, kernel_size=(3, 3), padding=1, dilation=2),
            nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=1)
        ).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)

        self.context_aggregation = ContextAggregationNetwork(out_channels * 8).to(device)
        self.flatten = nn.Flatten().to(device)

        self.n_mels = n_mels
        self.target_length = target_length
        self.num_stems = num_stems

        self.fc1 = None
        self.fc2 = nn.Linear(512, self.n_mels * self.target_length * self.num_stems).to(device)
        nn.init.xavier_normal_(self.fc2.weight)

    def get_cache_key(self, segment_idx, stage):
        """Generate a unique cache key based on the input data."""
        data_string = f"{segment_idx}_{stage}"
        return hashlib.md5(data_string.encode()).hexdigest()

    def load_cache_index(self):
        if os.path.exists(self.index_file_path):
            try:
                with open(self.index_file_path, 'r') as f:
                    self.cache_index = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.error("Cache index file is corrupted. Rebuilding index.")
                self.cache_index = defaultdict(list)
                self.rebuild_cache_index()
        else:
            self.cache_index = defaultdict(list)
            self.rebuild_cache_index()

    def save_cache_index(self):
        with open(self.index_file_path, 'w') as f:
            json.dump(self.cache_index, f)

    def rebuild_cache_index(self):
        try:
            with h5py.File(self.cache_file_path, 'a') as f:
                for key in f.keys():
                    segment_idx, stage = key.split('_')
                    self.cache_index[key] = (int(segment_idx), stage)
            self.save_cache_index()
        except Exception as e:
            logger.error(f"Error rebuilding cache index: {e}")

    def save_fixed_length_string(self, data, file_name, dataset_name, length=100):
        with h5py.File(file_name, 'a') as f:
            dtype = 'S{}'.format(length)  # Fixed length string
            if dataset_name in f:
                del f[dataset_name]
            dset = f.create_dataset(dataset_name, (len(data),), dtype=dtype)
            dset[:] = [d.encode('utf-8')[:length] for d in data]

    def sanitize_data(self, data):
        if isinstance(data, str):
            return data.replace('\x00', '')  # Remove NULL characters
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.sanitize_data(value) for key, value in data.items()}
        return data

    def save_intermediate(self, x, segment_idx, stage):
        key = self.get_cache_key(segment_idx, stage)
        try:
            if x.numel() > 1e6:  # If data is too large, save in JSON
                data = x.cpu().detach().numpy().tolist()
                with open(self.json_cache_file_path, 'a') as f:
                    json.dump({key: data}, f)
            else:
                sanitized_data = self.sanitize_data(x.cpu().detach().numpy().astype(np.float32).tobytes())
                self.save_fixed_length_string(sanitized_data, self.cache_file_path, key)
            self.cache_index[key] = x.cpu().detach().numpy().shape
            self.save_cache_index()
        except Exception as e:
            logger.error(f"Error saving intermediate data: {e}")
            self.handle_cache_error()

    def load_intermediate(self, segment_idx, stage):
        key = self.get_cache_key(segment_idx, stage)
        if key in self.cache_index:
            try:
                if os.path.exists(self.json_cache_file_path):
                    with open(self.json_cache_file_path, 'r') as f:
                        data = json.load(f).get(key)
                        if data:
                            return torch.tensor(data, device=self.device)
                else:
                    with h5py.File(self.cache_file_path, 'r') as f:
                        data = np.frombuffer(f[key][:], dtype=np.float32)
                        data = torch.from_numpy(data).view(self.cache_index[key]).to(self.device)
                        return data
            except Exception as e:
                logger.error(f"Error loading intermediate data: {e}")
                self.handle_cache_error()
        return None

    def handle_cache_error(self):
        if os.path.exists(self.cache_file_path):
            os.remove(self.cache_file_path)
        if os.path.exists(self.json_cache_file_path):
            os.remove(self.json_cache_file_path)
        self.rebuild_cache_index()

    def clear_intermediate(self):
        if os.path.exists(self.cache_file_path):
            os.remove(self.cache_file_path)
        if os.path.exists(self.json_cache_file_path):
            os.remove(self.json_cache_file_path)
        self.cache_index.clear()
        self.save_cache_index()

    @cached(cache=LRUCache(maxsize=100))
    def preprocess_and_cache(self, x, segment_idx, stage):
        cached_data = self.load_intermediate(segment_idx, stage)
        if cached_data is not None:
            return cached_data.to(self.device)
        
        with torch.no_grad():
            x = x.to(self.device)
            logger.info(f"Segment {segment_idx}, Stage {stage}: Input shape: {x.shape}")
            try:
                if stage == 'conv1':
                    x = self.conv1(x)
                    assert x.shape[1] == self.conv1.pointwise.out_channels, f"Expected {self.conv1.pointwise.out_channels} channels after conv1, got {x.shape[1]}"
                elif stage == 'pool1':
                    x = self.pool1(x)
                elif stage == 'res_block1':
                    x = self.res_block1(x)
                elif stage == 'attention1':
                    x = self.attention1(x)
                elif stage == 'conv2':
                    x = self.conv2(x)
                    assert x.shape[1] == self.conv2[1].out_channels, f"Expected {self.conv2[1].out_channels} channels after conv2, got {x.shape[1]}"
                elif stage == 'pool2':
                    x = self.pool2(x)
                elif stage == 'res_block2':
                    x = self.res_block2(x)
                elif stage == 'attention2':
                    x = self.attention2(x)
                elif stage == 'conv3':
                    x = self.pad_if_needed(x, self.conv3[0].kernel_size, self.conv3[0].dilation[0])
                    x = self.conv3(x)
                    assert x.shape[1] == self.conv3[1].out_channels, f"Expected {self.conv3[1].out_channels} channels after conv3, got {x.shape[1]}"
                elif stage == 'pool3':
                    x = self.pool3(x)
                elif stage == 'conv4':
                    x = self.pad_if_needed(x, self.conv4[0].kernel_size, self.conv4[0].dilation[0])
                    x = self.conv4(x)
                    assert x.shape[1] == self.conv4[1].out_channels, f"Expected {self.conv4[1].out_channels} channels after conv4, got {x.shape[1]}"
                elif stage == 'pool4':
                    x = self.pool4(x)
                elif stage == 'context_aggregation':
                    x = self.context_aggregation(x)
            except Exception as e:
                logger.error(f"Error in preprocess_and_cache (segment {segment_idx}, stage {stage}): {type(e).__name__} - {e}")
                raise
            logger.info(f"Segment {segment_idx}, Stage {stage}: Output shape: {x.shape}")

            self.save_intermediate(x, segment_idx, stage)
        return x

    def pad_if_needed(self, x, kernel_size, dilation=1):
        min_size = (
            (kernel_size[0] - 1) * dilation + 1,
            (kernel_size[1] - 1) * dilation + 1,
        )
        if x.size(2) < min_size[0] or x.size(3) < min_size[1]:
            padding = [
                0, max(0, min_size[1] - x.size(3)),
                0, max(0, min_size[0] - x.size(2))
            ]
            x = F.pad(x, padding, mode='constant')
        return x

    def save_concat_output(self, x, stage):
        try:
            if x.numel() > 1e6:  # If data is too large, save in JSON
                data = x.cpu().detach().numpy().tolist()
                with open(self.json_cache_file_path, 'a') as f:
                    json.dump({f"{stage}_output": data}, f)
            else:
                sanitized_data = self.sanitize_data(x.cpu().detach().numpy().astype(np.float32).tobytes())
                self.save_fixed_length_string(sanitized_data, self.cache_file_path, f"{stage}_output")
            self.cache_index[f"{stage}_output"] = x.cpu().detach().numpy().shape
            self.save_cache_index()
        except Exception as e:
            logger.error(f"Error saving concatenated output: {e}")
            self.handle_cache_error()

    def load_concat_output(self, expected_size):
        dset_name = "concat_output"
        try:
            if os.path.exists(self.json_cache_file_path):
                with open(self.json_cache_file_path, 'r') as f:
                    data = json.load(f).get(dset_name)
                    if data and len(data) == expected_size:
                        return torch.tensor(data, device=self.device)
            else:
                with h5py.File(self.cache_file_path, 'r') as f:
                    if dset_name in f and f[dset_name].shape[1] == expected_size:
                        data = np.frombuffer(f[dset_name][:], dtype=np.float32)
                        return torch.from_numpy(data).view(expected_size).to(self.device)
        except Exception as e:
            logger.error(f"Error loading concatenated output: {e}")
            self.handle_cache_error()
        return None

    def forward(self, x, suppress_reading_messages=True, initialize=False):
        if not suppress_reading_messages:
            logger.info(f"Input shape: {x.shape}")

        x = x.to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        if not suppress_reading_messages:
            logger.info(f"Shape after adding dimensions: {x.shape}")

        # Ensure preprocess_and_cache is correctly cached
        dynamic_cache_management(self.preprocess_and_cache.cache)

        if initialize:
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.res_block1(x)
            x = self.attention1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.res_block2(x)
            x = self.attention2(x)
            x = self.pad_if_needed(x, self.conv3[0].kernel_size, self.conv3[0].dilation[0])
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.pad_if_needed(x, self.conv4[0].kernel_size, self.conv4[0].dilation[0])
            x = self.conv4(x)
            x = self.pool4(x)
            x = self.pool5(x)
            x = self.context_aggregation(x)
            x = self.flatten(x)
            return x

        kernel_size = self.conv1.kernel_size[0]
        dilation = self.conv1.dilation[0]
        min_segment_length = (kernel_size - 1) * dilation + 1
        segment_length = max(x.shape[-2] // 10, min_segment_length)

        num_segments = (x.shape[-2] + segment_length - 1) // segment_length
        segments = [x[:, :, i*segment_length:(i+1)*segment_length, :] for i in range(num_segments)]

        outputs = []
        for i, segment in enumerate(segments):
            if not suppress_reading_messages:
                logger.info(f"Shape of segment {i+1} before processing: {segment.shape}")

            try:
                segment = segment.to(self.device)
                segment = self.preprocess_and_cache(segment, i, 'conv1')
                segment = self.preprocess_and_cache(segment, i, 'pool1')
                segment = self.preprocess_and_cache(segment, i, 'res_block1')
                segment = self.preprocess_and_cache(segment, i, 'attention1')
                segment = self.preprocess_and_cache(segment, i, 'conv2')
                segment = self.preprocess_and_cache(segment, i, 'pool2')
                segment = self.preprocess_and_cache(segment, i, 'res_block2')
                segment = self.preprocess_and_cache(segment, i, 'attention2')
                segment = self.pad_if_needed(segment, self.conv3[0].kernel_size, self.conv3[0].dilation[0])
                segment = self.preprocess_and_cache(segment, i, 'conv3')
                segment = self.pad_if_needed(segment, self.conv4[0].kernel_size, self.conv4[0].dilation[0])
                segment = self.preprocess_and_cache(segment, i, 'conv4')
                segment = self.preprocess_and_cache(segment, i, 'pool4')
                segment = self.preprocess_and_cache(segment, i, 'context_aggregation')
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {type(e).__name__} - {e}")
                raise

            if not suppress_reading_messages:
                logger.info(f"Shape of segment {i+1} after processing: {segment.shape}")

            segment = self.flatten(segment)
            outputs.append(segment)  # No need to move to CPU here
        
        x = torch.cat(outputs, dim=0).to(self.device)  # Concatenate along the batch dimension (0)

        logger.info(f"Shape of x after concatenation: {x.shape}")

        fc_input_size = x.shape[1]

        cached_data = self.load_concat_output(fc_input_size)
        if cached_data is not None:
            x = cached_data.to(self.device)
        else:
            self.save_concat_output(x, 'concat')

        if self.fc1 is None or self.fc1.in_features != fc_input_size or initialize:
            self.fc1 = nn.Linear(fc_input_size, 512).to(self.device)
            nn.init.xavier_normal_(self.fc1.weight)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)
        x = x.view(-1, self.num_stems, self.n_mels, self.target_length)

        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=512, out_channels=64, n_mels=128, target_length=256, device="cpu"):
        super(KANDiscriminator, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.InstanceNorm2d(64).to(device)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.InstanceNorm2d(128).to(device)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1).to(device)
        self.bn3 = nn.InstanceNorm2d(out_channels).to(device)

        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv3 = nn.utils.spectral_norm(self.conv3)

        self.fc1 = None

    def _forward_conv_layers(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 1).to(self.device)
            nn.init.xavier_normal_(self.fc1.weight)
        elif x.shape[1] != self.fc1.in_features:
            logger.warning("Flattened input size doesn't match fc1. Skipping this batch.")
            return None

        x = torch.sigmoid(self.fc1(x))
        return x

def convert_to_3_channels(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 2:
        x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
    elif x.shape[1] > 3:
        x = x[:, :3, :, :]
    return x

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, x, target):
        x_features = self.feature_extractor(x)
        target_features = self.feature_extractor(target)
        loss = self.criterion(x_features, target_features)
        return loss

def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = alpha * real_data + ((1 - alpha) * fake_data)
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

@cached(cache=LRUCache(maxsize=100))
def extract_and_cache_features(input_tensor, feature_extractor):
    return feature_extractor(input_tensor)

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device=None, freeze_fc_layers=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, "./cache", device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.train()

    if freeze_fc_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

    model.to(device)
    return model

def dynamic_cache_management(preprocess_and_cache):
    if torch.cuda.is_available():
        reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3
        allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
        physical_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        if reserved_memory > 8.0 or allocated_memory > 8.0:
            preprocess_and_cache.clear()
            new_cache_size = max(preprocess_and_cache.maxsize - 10, 10)
        else:
            new_cache_size = min(preprocess_and_cache.maxsize + 10, 100)
        
        preprocess_and_cache = LRUCache(maxsize=new_cache_size)
        logger.info(f'Cache size adjusted to: {new_cache_size}')

    dynamic_max_split_size()

def dynamic_max_split_size():
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    max_split_size_mb = int(total_memory * 0.5)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb={max_split_size_mb}'
    logger.info(f'Set PYTORCH_CUDA_ALLOC_CONF to max_split_size_mb={max_split_size_mb}')

def check_and_reshape(outputs, targets, logger):
    target_numel = targets.numel()
    output_numel = outputs.numel()

    if output_numel != target_numel:
        logger.warning(f"Output size ({output_numel}) does not match the expected size ({target_numel}). Attempting to reshape.")
        
        outputs = outputs.view(-1)
        
        if output_numel > target_numel:
            outputs = outputs[:target_numel]
        else:
            padding_size = target_numel - output_numel
            outputs = F.pad(outputs, (0, padding_size))
        
        outputs = outputs.view(targets.shape)

    return outputs

def main():
    logging.info("KANWithDepthwiseConv model script executed")

if __name__ == "__main__":
    main()
