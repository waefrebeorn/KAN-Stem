import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch.optim as optim
from utils import detect_parameters
from model import KANWithDepthwiseConv, KANDiscriminator
from cached_dataset import StemSeparationDataset
from train_single_stem import start_training
from log_setup import logger  # Import the logger from the log_setup module

# Hook to log when a tensor is allocated
def allocation_hook(tensor):
    logger.info(f"Allocated tensor of size {tensor.size()} with memory {tensor.element_size() * tensor.nelement() / 1024 ** 2:.2f} MB")

# Register hooks
original_new = torch.Tensor.__new__
def new_tensor(cls, *args, **kwargs):
    tensor = original_new(cls, *args, **kwargs)
    allocation_hook(tensor)
    return tensor

torch.Tensor.__new__ = new_tensor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train KAN models for stem separation.")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for training.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--save-interval', type=int, default=1, help='Interval for saving checkpoints.')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps.')
    parser.add_argument('--num-stems', type=int, default=7, help='Number of stems to separate.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers.')
    parser.add_argument('--cache-dir', type=str, default='./cache', help='Cache directory for storing intermediate data.')

    args = parser.parse_args()
    start_training(args.data_dir, args.batch_size, args.num_epochs, args.learning_rate, args.use_cuda, args.checkpoint_dir, args.save_interval, args.accumulation_steps, args.num_stems, args.num_workers, args.cache_dir)
