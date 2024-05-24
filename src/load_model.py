import torch
from train import KANWithDepthwiseConv  # Import here to avoid circular imports

def load_model(checkpoint_path, in_channels, out_channels, n_mels, target_length):
    model = KANWithDepthwiseConv(in_channels, out_channels, n_mels, target_length)
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model
