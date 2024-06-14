import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import h5py
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

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

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, groups=channels, bias=False)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        x += residual
        return F.relu(x)

class ContextAggregationNetwork(nn.Module):
    def __init__(self, channels):
        super(ContextAggregationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems, device, channel_multiplier=1.0):
        super(KANWithDepthwiseConv, self).__init__()

        self.device = device
        self.num_stems = num_stems
        self.n_mels = n_mels
        self.target_length = target_length

        self.conv1 = nn.Conv2d(in_channels, int(out_channels * channel_multiplier), kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block1 = ResidualBlock(int(out_channels * channel_multiplier), dilation_rate=1)
        self.attention1 = AttentionLayer(int(out_channels * channel_multiplier))

        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier), kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(int(out_channels * 2 * channel_multiplier)),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_block2 = ResidualBlock(int(out_channels * 2 * channel_multiplier), dilation_rate=2)
        self.attention2 = AttentionLayer(int(out_channels * 2 * channel_multiplier))

        self.conv3 = nn.Sequential(
            nn.Conv2d(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier), kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(int(out_channels * 4 * channel_multiplier)),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(out_channels * 4 * channel_multiplier), int(out_channels * 8 * channel_multiplier), kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(int(out_channels * 8 * channel_multiplier)),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.context_aggregation = ContextAggregationNetwork(int(out_channels * 8 * channel_multiplier))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(int(out_channels * 8 * 2 * 2 * channel_multiplier), 512)
        self.fc2 = nn.Linear(512, n_mels * target_length * num_stems)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward_impl(self, x, suppress_reading_messages=False):
        layers = [
            self.conv1, self.pool1, self.res_block1, self.attention1, 
            self.conv2, self.pool2, self.res_block2, self.attention2, 
            self.conv3, self.pool3, self.conv4, self.pool4, 
            self.pool5, self.context_aggregation, self.flatten
        ]
        for layer in layers:
            try:
                x = checkpoint(layer, x)
            except Exception as e:
                logger.error(f"Error in layer {layer.__class__.__name__}: {e}")
                raise

            if not suppress_reading_messages:
                logger.info(f"{layer.__class__.__name__}: {x.shape}")
        return x

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

        if initialize:
            x = self.forward_impl(x, suppress_reading_messages)
            return x

        kernel_size = self.conv1.kernel_size[0]
        dilation = self.conv1.dilation[0]
        min_segment_length = max((kernel_size - 1) * dilation + 1, self.target_length)
        if x.shape[-1] > min_segment_length:
            segment_length = max(min_segment_length, x.shape[-1] // 10)
            num_segments = (x.shape[-1] + segment_length - 1) // segment_length
            segments = [x[:, :, :, i*segment_length:(i+1)*segment_length] for i in range(num_segments)]
        else:
            segments = [x]

        if not suppress_reading_messages:
            logger.info(f"Number of segments: {len(segments)}")

        outputs = []
        for i, segment in enumerate(segments):
            if not suppress_reading_messages:
                logger.info(f"Shape of segment {i+1} before processing: {segment.shape}")
            
            if segment.numel() == 0:
                logger.error(f"Segment {i+1} is empty and cannot be processed.")
                continue
            try:
                segment_output = self.forward_impl(segment, suppress_reading_messages)
                outputs.append(segment_output)
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {e}")
                raise

        if not outputs:
            logger.error("No segments were processed successfully.")
            return None

        x = torch.cat(outputs, dim=-1)  # Concatenate along the time dimension

        total_length = x.shape[-1]  # Total length after concatenation

        x = x.view(-1, self.num_stems, self.n_mels, total_length)
        logger.info(f"Shape of x after concatenation: {x.shape}")

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)

        x = x.view(-1, self.num_stems, self.n_mels, total_length)
        logger.info(f"Shape of final output: {x.shape}")

        return x

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, n_mels=128, target_length=256, device="cpu", channel_multiplier=1.0):
        super(KANDiscriminator, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(in_channels, int(out_channels * channel_multiplier), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(int(out_channels * channel_multiplier))
        self.conv2 = nn.Conv2d(int(out_channels * channel_multiplier), int(out_channels * 2 * channel_multiplier), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(int(out_channels * 2 * channel_multiplier))
        self.conv3 = nn.Conv2d(int(out_channels * 2 * channel_multiplier), int(out_channels * 4 * channel_multiplier), kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(int(out_channels * 4 * channel_multiplier))

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

        if self.fc1 is None or self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 1).to(self.device)
            nn.init.xavier_normal_(self.fc1.weight)

        x = torch.sigmoid(self.fc1(x))
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

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device=None, freeze_fc_layers=False, channel_multiplier=1.0):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, device, channel_multiplier)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.train()

    if freeze_fc_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

    model.to(device)
    return model

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

def load_from_cache(cache_file_path):
    with h5py.File(cache_file_path, 'r') as f:
        input_data = torch.tensor(f['input'][:]).float()
        target_data = torch.tensor(f['target'][:]).float()
    return {'input': input_data, 'target': target_data}

if __name__ == "__main__":
    logger.info("Model script executed directly.")
else:
    logger.info("Model script imported as a module.")
