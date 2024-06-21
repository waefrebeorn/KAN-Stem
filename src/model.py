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

class MemoryEfficientStemSeparationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_mels=64, target_length=22050):
        super(MemoryEfficientStemSeparationModel, self).__init__()
        self.n_mels = n_mels
        self.target_length = target_length

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 16, kernel_size=3, stride=1, padding=1),
            self.conv_block(16, 32, kernel_size=3, stride=2, padding=1),
            self.conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            self.conv_block(64, 128, kernel_size=3, stride=2, padding=1)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            self.conv_block(128, 64, kernel_size=3, stride=1, padding=1),
            self.conv_block(64, 32, kernel_size=3, stride=1, padding=1),
            self.conv_block(32, 16, kernel_size=3, stride=1, padding=1)
        ])
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # Ensure input is 4D (batch_size, channels, n_mels, target_length)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze(2)

        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = checkpoint(encoder_layer, x)
            encoder_outputs.append(x)
            # Free up memory
            if len(encoder_outputs) > 1:
                del encoder_outputs[-2]
            torch.cuda.empty_cache()

        # Upsample
        x = F.interpolate(x, size=(self.n_mels, self.target_length), mode='bilinear', align_corners=False)

        # Decoder
        for decoder_layer in self.decoder:
            x = checkpoint(decoder_layer, x)
            torch.cuda.empty_cache()

        # Final convolution
        x = self.final_conv(x)

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
        if x.dim() == 5:
            x = x.squeeze(2)  # Remove the singleton dimension if needed

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

    model = MemoryEfficientStemSeparationModel(in_channels, out_channels, n_mels, target_length).to(device)
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
