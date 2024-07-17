import os
import torch
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, epoch, segment_idx, model, optimizer_g, optimizer_d, scaler_g, scaler_d, model_params, training_params, stem_name, is_epoch_checkpoint=True):
    checkpoint = {
        'epoch': epoch,
        'segment_idx': segment_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scaler_g_state_dict': scaler_g.state_dict(),
        'scaler_d_state_dict': scaler_d.state_dict(),
        'n_mels': model_params['n_mels'],
        'target_length': model_params['target_length'],
        'optimizer_name_g': model_params['optimizer_name_g'],
        'optimizer_name_d': model_params['optimizer_name_d'],
        'initial_lr_g': training_params['initial_lr_g'],
        'initial_lr_d': training_params['initial_lr_d'],
        'weight_decay': training_params['weight_decay']
    }
    if is_epoch_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, f'stem_{stem_name}_epoch_{epoch}.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'stem_{stem_name}_epoch_{epoch}_segment_{segment_idx}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint
