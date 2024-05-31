import torch
import torch.nn as nn

data_dir = "path_to_data_dir"
val_dir = "path_to_validation_dir"
batch_size = 32
num_epochs = 10
learning_rate_g = 0.001
learning_rate_d = 0.00005
use_cuda = True
checkpoint_dir = "path_to_checkpoint_dir"
save_interval = 1
accumulation_steps = 1
num_stems = 7
num_workers = 4
cache_dir = "path_to_cache_dir"
loss_function_g = nn.L1Loss()
loss_function_d = nn.BCEWithLogitsLoss()
optimizer_name_g = "Adam"
optimizer_name_d = "Adam"
perceptual_loss_flag = True
clip_value = 1.0
scheduler_step_size = 5
scheduler_gamma = 0.5
tensorboard_flag = True
apply_data_augmentation = False
add_noise = False
noise_amount = 0.1
early_stopping_patience = 3
weight_decay = 1e-4
suppress_messages = False
suppress_reading_messages = False

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
