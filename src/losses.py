import torch
import torch.nn.functional as F

def si_snr_loss(pred, target, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    s_target = torch.sum(pred * target, dim=-1, keepdim=True) * target / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = pred - s_target
    si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps) + eps)
    return -si_snr.mean()
