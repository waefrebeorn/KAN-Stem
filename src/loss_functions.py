import torch
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def wasserstein_loss(y_pred, y_true):
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must be tensors, but got None")
    return torch.mean(y_true * y_pred)

def mse_loss(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)

def l1_loss(y_pred, y_true):
    return torch.nn.functional.l1_loss(y_pred, y_true)

def smooth_l1_loss(y_pred, y_true):
    return torch.nn.functional.smooth_l1_loss(y_pred, y_true)

def bce_with_logits_loss(y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
