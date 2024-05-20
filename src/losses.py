import torch
import torch.nn.functional as F

def si_snr_loss(preds, target):
    def pairwise_dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)
    
    def l2_norm(x):
        return torch.norm(x, dim=-1, keepdim=True)
    
    target_dot = pairwise_dot(target, target)
    proj = target_dot / (pairwise_dot(target, target) + 1e-8) * target
    e_noise = preds - proj
    
    snr = pairwise_dot(proj, proj) / (pairwise_dot(e_noise, e_noise) + 1e-8)
    si_snr = 10 * torch.log10(snr + 1e-8)
    
    return -torch.mean(si_snr)
