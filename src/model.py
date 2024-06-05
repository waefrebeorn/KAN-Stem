import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

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
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(ResidualBlock, self).__init__()
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in dilation_rates])
        self.dropout = nn.Dropout(p=0.3)  # Dropout regularization

    def forward(self, x):
        residual = x
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu_(bn(conv(x)))
        x = self.dropout(x)  # Apply dropout
        x += residual
        return F.relu_(x)

class ContextAggregationNetwork(nn.Module):
    def __init__(self, channels):
        super(ContextAggregationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, suppress_reading_messages=False):
        if not suppress_reading_messages:
            logger.info("Shape before conv1: %s", x.shape)
        x = F.relu_(self.bn1(self.conv1(x)))
        if not suppress_reading_messages:
            logger.info("Shape after conv1: %s", x.shape)
        x = F.relu_(self.bn2(self.conv2(x)))
        if not suppress_reading_messages:
            logger.info("Shape after conv2: %s", x.shape)
        return x

class KANWithDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_mels, target_length, num_stems, cache_dir, device):
        super(KANWithDepthwiseConv, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=(3, 3), padding=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.res_block1 = ResidualBlock(out_channels, out_channels).to(device)
        self.attention1 = AttentionLayer(out_channels)  # Add attention layer
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels * 2, kernel_size=(3, 3), padding=1).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.res_block2 = ResidualBlock(out_channels * 2, out_channels * 2).to(device)
        self.attention2 = AttentionLayer(out_channels * 2)  # Add attention layer
        self.conv3 = DepthwiseSeparableConv(out_channels * 2, out_channels * 4, kernel_size=(3, 3), padding=1, dilation=2).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.conv4 = DepthwiseSeparableConv(out_channels * 4, out_channels * 8, kernel_size=(3, 3), padding=1, dilation=2).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)

        self.context_aggregation = ContextAggregationNetwork(out_channels * 8).to(device)
        self.flatten = nn.Flatten().to(device)

        self.n_mels = n_mels
        self.target_length = target_length
        self.num_stems = num_stems
        self.cache_dir = cache_dir
        self.device = device

        # Initialize fully connected layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, n_mels, target_length, device=device)
            dummy_output = self.pool4(self.conv4(self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))))
            dummy_output = self.pool5(dummy_output)
            dummy_output = self.context_aggregation(dummy_output, suppress_reading_messages=True)  # Added context_aggregation
            dummy_output = self.flatten(dummy_output)
            fc_input_size = dummy_output.shape[1]
        self.fc1 = nn.Linear(fc_input_size, 1024).to(device)
        nn.init.xavier_normal_(self.fc1.weight)  # Initialize fc1 weights
        self.fc2 = nn.Linear(1024, self.n_mels * self.target_length * self.num_stems).to(device)
        nn.init.xavier_normal_(self.fc2.weight)  # Initialize fc2 weights

    def forward(self, x, suppress_reading_messages=False):
        x = x.to(self.device)  # Ensure x is on the correct device
        if x.dim() == 2:  # Add channel and batch dimensions if not present
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:  # Add channel dimension if not present
            x = x.unsqueeze(1)
        if not suppress_reading_messages:
            logger.info("Shape before conv1: %s", x.shape)
        x = F.relu_(self.conv1(x))
        if not suppress_reading_messages:
            logger.info("Shape after conv1: %s", x.shape)
        x = self.pool1(x)
        x = self.res_block1(x)  # Add residual block
        x = self.attention1(x)  # Apply attention

        x = F.relu_(self.conv2(x))
        x = self.pool2(x)
        x = self.res_block2(x)  # Add residual block
        x = self.attention2(x)  # Apply attention

        x = F.relu_(self.conv3(x))
        x = self.pool3(x)
        x = F.relu_(self.conv4(x))
        x = self.pool4(x)
        x = self.pool5(x)

        if not suppress_reading_messages:
            logger.info("Shape before context_aggregation: %s", x.shape)
        x = self.context_aggregation(x, suppress_reading_messages=suppress_reading_messages)
        if not suppress_reading_messages:
            logger.info("Shape after context_aggregation: %s", x.shape)

        x = self.flatten(x)
        if not suppress_reading_messages:
            logger.info("Shape after flatten: %s", x.shape)

        x = F.relu_(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)  # Use tanh activation to restrict output to [-1, 1]
        x = x.view(-1, self.num_stems, self.n_mels, self.target_length)
        if not suppress_reading_messages:
            logger.info("Final output shape: %s", x.shape)

        return x

    def cache_activation(self, x, name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_metadata_path = os.path.join(self.cache_dir, "cache_metadata.pt")

        # Load existing cache metadata
        cache_metadata = {}
        if os.path.exists(cache_metadata_path):
            try:
                cache_metadata = torch.load(cache_metadata_path)
            except (FileNotFoundError, EOFError, RuntimeError) as e:
                logger.error(f"Error loading cache metadata file: {e}")
                cache_metadata = {}

        # Store the activation tensor in the cache metadata dictionary
        cache_metadata[name] = x.detach().cpu()

        # Save the updated cache metadata
        try:
            temp_cache_path = f"{cache_metadata_path}.tmp"
            torch.save(cache_metadata, temp_cache_path)
            os.replace(temp_cache_path, cache_metadata_path)
        except Exception as e:
            logger.error(f"Error saving cache metadata file: {e}")

        return cache_metadata_path

class KANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, n_mels=128, target_length=256, device="cpu"):
        super(KANDiscriminator, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.InstanceNorm2d(64).to(device)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.InstanceNorm2d(128).to(device)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1).to(device)
        self.bn3 = nn.InstanceNorm2d(out_channels).to(device)

        # Apply spectral normalization to conv layers
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv3 = nn.utils.spectral_norm(self.conv3)

        # Calculate the flattened size based on the convolutional layers' output
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, n_mels, target_length, device=device)
            dummy_output = self._forward_conv_layers(dummy_input)
            flattened_size = dummy_output.view(-1).shape[0]

        # Modify the input size of fc1 to match the calculated flattened size
        self.fc1 = nn.Linear(flattened_size, 1).to(device)
        nn.init.xavier_normal_(self.fc1.weight)  # Initialize fc1 weights

    def _forward_conv_layers(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.sigmoid(self.fc1(x))
        return x

def convert_to_3_channels(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 2:
        x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
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

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length, num_stems, device=None, freeze_fc_layers=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatic device selection

    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length, num_stems, "./cache", device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.train()  # Switch to training mode

    if freeze_fc_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

    model.to(device)
    return model
