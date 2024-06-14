import torch
import torch.optim as optim
from model import KANWithDepthwiseConv, KANDiscriminator
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def create_model_and_optimizer(device, n_mels, target_length, initial_lr_g, initial_lr_d, optimizer_name_g, optimizer_name_d, weight_decay, channel_multiplier):
    # Create the generator model
    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=2, device=device, channel_multiplier=channel_multiplier).to(device)
    
    # Create the discriminator model
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, device=device, channel_multiplier=channel_multiplier).to(device)
    
    # Correct optimizer names
    if optimizer_name_g.lower() == 'rmsprop':
        optimizer_name_g = 'RMSprop'
    if optimizer_name_d.lower() == 'rmsprop':
        optimizer_name_d = 'RMSprop'
    
    # Create the optimizers for both models
    optimizer_g = getattr(optim, optimizer_name_g)(model.parameters(), lr=initial_lr_g, weight_decay=weight_decay)
    optimizer_d = getattr(optim, optimizer_name_d)(discriminator.parameters(), lr=initial_lr_d, weight_decay=weight_decay)
    
    return model, discriminator, optimizer_g, optimizer_d

def initialize_model(device, n_mels, target_length):
    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=1, device=device).to(device)
    return model

def initialize_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    return get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay)

if __name__ == "__main__":
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
