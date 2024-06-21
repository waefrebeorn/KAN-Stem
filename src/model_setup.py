import torch
import torch.optim as optim
from model import MemoryEfficientStemSeparationModel, KANDiscriminator
from utils import get_optimizer
from torch.cuda.amp import GradScaler
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def create_model_and_optimizer(device, n_mels, target_length, initial_lr_g, initial_lr_d, 
                               optimizer_name_g, optimizer_name_d, weight_decay):
    # Create the generator model
    model = MemoryEfficientStemSeparationModel(in_channels=1, out_channels=1, n_mels=n_mels, 
                         target_length=target_length).to(device)

    # Create the discriminator model
    discriminator = KANDiscriminator(in_channels=1, out_channels=32, n_mels=n_mels, 
                                     target_length=target_length, device=device).to(device)

    # Create the optimizers
    optimizer_g = get_optimizer(optimizer_name_g, model.parameters(), initial_lr_g, weight_decay)
    optimizer_d = get_optimizer(optimizer_name_d, discriminator.parameters(), initial_lr_d, weight_decay)

    # Create gradient scalers for mixed precision training
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    return model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d

def initialize_model(device, n_mels, target_length):
    """Initializes the generator model only (no discriminator)."""
    model = MemoryEfficientStemSeparationModel(in_channels=1, out_channels=1, n_mels=n_mels, 
                         target_length=target_length).to(device)
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

    model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = create_model_and_optimizer(
        device, n_mels, target_length, learning_rate_g, learning_rate_d,
        optimizer_name_g, optimizer_name_d, weight_decay
    )

    print(f"Model: {model}")
    print(f"Discriminator: {discriminator}")
    print(f"Optimizer G: {optimizer_g}")
    print(f"Optimizer D: {optimizer_d}")
    print(f"Scaler G: {scaler_g}")
    print(f"Scaler D: {scaler_d}")
