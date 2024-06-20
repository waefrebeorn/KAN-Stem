import torch
import torch.optim as optim
from model import KANWithBSRBF, KANDiscriminator  # Updated import for KANWithBSRBF
from utils import get_optimizer  # Import your get_optimizer function
import warnings

# Filter out development warnings
warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def create_model_and_optimizer(device, n_mels, target_length, initial_lr_g, initial_lr_d, 
                                optimizer_name_g, optimizer_name_d, weight_decay, channel_multiplier):

    # Create the generator model (single-channel input for mel spectrograms)
    model = KANWithBSRBF(in_channels=3, out_channels=64, n_mels=n_mels, 
                         target_length=target_length, device=device, 
                         channel_multiplier=channel_multiplier).to(device)

    # Create the discriminator model (multi-channel input for stacked features)
    discriminator = KANDiscriminator(in_channels=3, out_channels=64, n_mels=n_mels, 
                                     target_length=target_length, device=device, 
                                     channel_multiplier=channel_multiplier).to(device)

    # Create the optimizers for the generator and discriminator, using your utils function
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), initial_lr_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), initial_lr_d, weight_decay)

    return model, discriminator, optimizer_g, optimizer_d

def initialize_model(device, n_mels, target_length):
    """Initializes the generator model only (no discriminator)."""
    model = KANWithBSRBF(in_channels=1, out_channels=64, n_mels=n_mels, 
                         target_length=target_length, device=device).to(device)
    return model

if __name__ == "__main__":
    # Example usage (optional, for testing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_mels = 128
    target_length = 256
    learning_rate_g = 1e-4
    learning_rate_d = 1e-4
    optimizer_name_g = 'Adam'
    optimizer_name_d = 'RMSprop'
    weight_decay = 1e-5
    channel_multiplier = 0.5

    model, discriminator, optimizer_g, optimizer_d = create_model_and_optimizer(
        device, n_mels, target_length, learning_rate_g, learning_rate_d,
        optimizer_name_g, optimizer_name_d, weight_decay, channel_multiplier
    )

    print(f"Model: {model}")
    print(f"Discriminator: {discriminator}")
    print(f"Optimizer G: {optimizer_g}")
    print(f"Optimizer D: {optimizer_d}")
