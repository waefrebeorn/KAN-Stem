import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import h5py
from torch.utils.checkpoint import checkpoint
import math
import gc

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

    @torch.cuda.amp.autocast()
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class BSRBF_KANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_size=5, spline_order=3, base_activation=torch.nn.SiLU, grid_range=[-1.5, 1.5]):
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

    @torch.cuda.amp.autocast()
    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 3 and x.size(2) == self.input_dim
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :, :-1]) + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, :, 1:])
        assert bases.size() == (x.size(0), x.size(1), self.input_dim, self.grid_size + self.spline_order)
        return bases.contiguous()

    @torch.cuda.amp.autocast()
    def forward(self, x):
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

    @torch.cuda.amp.autocast()
    def forward(self, x):
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

    @torch.cuda.amp.autocast()
    def forward(self, x):
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

    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class KANWithBSRBF(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, n_mels=127, target_length=44036, num_stems=4, device="cuda", channel_multiplier=1.0, grid_size=5, spline_order=3):
        super(KANWithBSRBF, self).__init__()
        self.device = device
        self.num_stems = num_stems
        self.n_mels = n_mels
        self.target_length = target_length
        self.channel_multiplier = channel_multiplier

        self.conv1 = nn.Conv2d(in_channels, int(out_channels * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block1 = ResidualBlock(int(out_channels * channel_multiplier), dilation_rate=1, grid_size=grid_size, spline_order=spline_order).cuda()
        self.attention1 = AttentionLayer(int(out_channels * channel_multiplier)).cuda()

        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier), kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(int(out_channels * 2 * channel_multiplier)),
            nn.ReLU(inplace=True)
        ).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block2 = ResidualBlock(int(out_channels * 2 * channel_multiplier), dilation_rate=2, grid_size=grid_size, spline_order=spline_order).cuda()
        self.attention2 = AttentionLayer(int(out_channels * 2 * channel_multiplier)).cuda()

        self.conv3 = nn.Sequential(
            nn.Conv2d(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier), kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(int(out_channels * 4 * channel_multiplier)),
            nn.ReLU(inplace=True)
        ).cuda()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(out_channels * 4 * channel_multiplier), int(out_channels * 8 * channel_multiplier), kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(int(out_channels * 8 * channel_multiplier)),
            nn.ReLU(inplace=True)
        ).cuda()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.context_aggregation = ContextAggregationNetwork(int(out_channels * 8 * channel_multiplier)).cuda()
        self.flatten = nn.Flatten()

        self.kan1 = None
        self.kan2 = None

    def adjust_kan_layers(self, input_size):
        if self.kan1 is None or self.kan1.input_dim != input_size:
            self.kan1 = BSRBF_KANLayer(input_size, 512, grid_size=5, spline_order=3).cuda()
            self.kan2 = BSRBF_KANLayer(512, 1 * self.n_mels * self.target_length, grid_size=5, spline_order=3).cuda()

    @torch.cuda.amp.autocast()
    def forward_impl(self, x, suppress_reading_messages=False):
        layers = [
            self.conv1, self.pool1, self.res_block1, self.attention1, 
            self.conv2, self.pool2, self.res_block2, self.attention2, 
            self.conv3, self.pool3, self.conv4, self.pool4, 
            self.pool5, self.context_aggregation, self.flatten
        ]
        for layer in layers:
            try:
                x = checkpoint(layer, x, use_reentrant=True)
            except Exception as e:
                logger.error(f"Error in layer {layer.__class__.__name__}: {e}")
                raise

            if not suppress_reading_messages:
                logger.info(f"{layer.__class__.__name__}: {x.shape}")
            
            # Strategic VRAM purging
            if layer in [self.pool2, self.pool4]:
                purge_vram()

        return x

    @torch.cuda.amp.autocast()
    def forward(self, x, suppress_reading_messages=True, is_initializing=False):
        if not suppress_reading_messages:
            logger.info(f"Input shape: {x.shape}")

        # Input Validation
        if x.dim() not in [2, 3, 4]:
            raise ValueError(f"Expected input tensor to have 2, 3, or 4 dimensions, got {x.dim()}")

        x = x.to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        if not suppress_reading_messages:
            logger.info(f"Shape after adding dimensions: {x.shape}")

        if is_initializing:
            x = self.forward_impl(x, suppress_reading_messages)
            return x

        if x.shape[-1] > self.target_length:
            num_segments = (x.shape[-1] + self.target_length - 1) // self.target_length
            segments = torch.split(x, self.target_length, dim=-1)
            if segments[-1].shape[-1] < self.target_length:
                segments = list(segments)
                segments[-1] = F.pad(segments[-1], (0, self.target_length - segments[-1].shape[-1]))
                segments = tuple(segments)
        else:
            if x.shape[-1] < self.target_length:
                x = F.pad(x, (0, self.target_length - x.shape[-1]))
            segments = (x,)

        if not suppress_reading_messages:
            logger.info(f"Number of segments: {len(segments)}")

        outputs = []
        for i, segment in enumerate(segments):
            if not suppress_reading_messages:
                logger.info(f"Shape of segment {i + 1} before processing: {segment.shape}")

            if segment.numel() == 0:
                logger.error(f"Segment {i + 1} is empty and cannot be processed.")
                continue
            try:
                segment_output = self.forward_impl(segment, suppress_reading_messages)
                outputs.append(segment_output)
            except Exception as e:
                logger.error(f"Error processing segment {i + 1}: {e}")
                raise

            # Strategic VRAM purging
            if i % 2 == 1:
                purge_vram()

        if not outputs:
            logger.error("No segments were processed successfully.")
            return None

        x = torch.cat(outputs, dim=0)

        # Dynamically adjust kan1 input size based on the actual input shape
        x = x.view(x.size(0), -1)
        self.adjust_kan_layers(x.shape[1])

        x = F.relu(self.kan1(x))
        x = self.kan2(x)

        # Adjust the shape here based on actual output
        try:
            output_shape = (-1, 1, self.n_mels, self.target_length)  # Updated to 1 channel
            x = x.view(*output_shape)
            logger.info(f"Shape of final output: {x.shape}")
        except RuntimeError as e:
            logger.error(f"Error reshaping tensor: {e}")
            logger.error(f"Input shape: {x.shape}")
            logger.error(f"Expected shape: {output_shape}")
            raise e

        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, n_mels=127, target_length=44036, device="cuda", channel_multiplier=1.0):
        super(KANDiscriminator, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(in_channels, int(out_channels * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn1 = nn.InstanceNorm2d(int(out_channels * channel_multiplier)).cuda()
        self.conv2 = nn.Conv2d(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn2 = nn.InstanceNorm2d(int(out_channels * 2 * channel_multiplier)).cuda()
        self.conv3 = nn.Conv2d(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier), kernel_size=3, padding=1, bias=False).cuda()
        self.bn3 = nn.InstanceNorm2d(int(out_channels * 4 * channel_multiplier)).cuda()

        self.fc1 = None

    @torch.cuda.amp.autocast()
    def _forward_conv_layers(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    @torch.cuda.amp.autocast()
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed
        x = x.to(self.device)  # Ensure the data is on the correct device
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.fc1 is None or self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 1).to(self.device)
            nn.init.xavier_normal_(self.fc1.weight)

        x = torch.sigmoid(self.fc1(x))
        return x

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor.cuda()
        self.criterion = nn.MSELoss().cuda()

    @torch.cuda.amp.autocast()
    def forward(self, x, target):
        x_features = self.feature_extractor(x)
        target_features = self.feature_extractor(target)
        loss = self.criterion(x_features, target_features)
        return loss

@torch.cuda.amp.autocast()
def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_data + ((1 - alpha) * fake_data)
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device=None, freeze_fc_layers=False, channel_multiplier=1.0):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KANWithBSRBF(in_channels, out_channels, n_mels, target_length, num_stems, device, channel_multiplier)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.train()

    if freeze_fc_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

    model.to(device)
    return model

@torch.cuda.amp.autocast()
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

@torch.cuda.amp.autocast()
def load_from_cache(cache_file_path, device='cuda'):
    try:
        with h5py.File(cache_file_path, 'r') as f:
            input_data = torch.tensor(f['input'][:], device=device).float()
            target_data = torch.tensor(f['target'][:], device=device).float()
            input_data = input_data.unsqueeze(0).unsqueeze(0) if input_data.dim() == 2 else input_data.unsqueeze(1)
            target_data = target_data.unsqueeze(0).unsqueeze(0) if target_data.dim() == 2 else target_data.unsqueeze(1)
        return {'input': input_data, 'target': target_data}
    except Exception as e:
        logger.error(f"Error loading batch from HDF5: {e}")
        raise

if __name__ == "__main__":
    logger.info("Model script executed directly.")
else:
    logger.info("Model script imported as a module.")
