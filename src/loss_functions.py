import torch

def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)

def mse_loss(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)

def l1_loss(y_pred, y_true):
    return torch.nn.functional.l1_loss(y_pred, y_true)

def smooth_l1_loss(y_pred, y_true):
    return torch.nn.functional.smooth_l1_loss(y_pred, y_true)

def bce_with_logits_loss(y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
