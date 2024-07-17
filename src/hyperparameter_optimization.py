import optuna
from train import start_training
import torch.nn as nn
from loss_functions import wasserstein_loss
from multiprocessing import Value
import hashlib
import warnings
import torch
from train import start_training_wrapper, stop_training_wrapper

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

def get_optimizer_scheduler(optimizer_name, optimizer_g, optimizer_d):
    if optimizer_name in ["SGD", "Momentum", "RMSProp"]:
        scheduler_g = torch.optim.lr_scheduler.CyclicLR(optimizer_g, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2', cycle_momentum=True)
        scheduler_d = torch.optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2', cycle_momentum=True)
    else:
        scheduler_g = torch.optim.lr_scheduler.CyclicLR(optimizer_g, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2', cycle_momentum=False)
        scheduler_d = torch.optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular2', cycle_momentum=False)
    return scheduler_g, scheduler_d

def calculate_accumulation_steps(effective_batch_size):
    accumulation_steps = effective_batch_size // 16
    if effective_batch_size % 16 != 0:
        accumulation_steps += 1
    return accumulation_steps

def objective_optuna(trial, gradio_params):
    # Respect the system call batch size
    effective_batch_size = gradio_params["batch_size"]
    accumulation_steps = calculate_accumulation_steps(effective_batch_size)

    learning_rate_g = trial.suggest_float('learning_rate_g', 1e-5, 1e-1, log=True)
    learning_rate_d = trial.suggest_float('learning_rate_d', 1e-5, 1e-1, log=True)
    perceptual_loss_weight = trial.suggest_float('perceptual_loss_weight', 0.0, 1.0)
    clip_value = trial.suggest_float('clip_value', 0.5, 1.5)

    stop_flag = Value('i', 0)  # Shared flag for stopping processes
    checkpoint_flag = Value('i', 0)  # Shared flag for saving checkpoints

    start_training_wrapper(
        data_dir=gradio_params["data_dir"],
        batch_size=effective_batch_size,  # Actual batch size from gradio_params
        num_epochs=5,  # Test with only 5 epochs
        learning_rate_g=learning_rate_g,
        learning_rate_d=learning_rate_d,
        use_cuda=gradio_params["use_cuda"],
        checkpoint_dir=gradio_params["checkpoint_dir"],
        save_interval=gradio_params["save_interval"],
        accumulation_steps=accumulation_steps,
        num_stems=gradio_params["num_stems"],
        num_workers=gradio_params["num_workers"],
        cache_dir=gradio_params["cache_dir"],
        loss_function_str_g="L1Loss",
        loss_function_str_d="WassersteinLoss",
        optimizer_name_g="Adam",
        optimizer_name_d="Adam",
        perceptual_loss_flag=True,
        perceptual_loss_weight=perceptual_loss_weight,
        clip_value=clip_value,
        scheduler_step_size=gradio_params["scheduler_step_size"],
        scheduler_gamma=gradio_params["scheduler_gamma"],
        tensorboard_flag=gradio_params["tensorboard_flag"],
        add_noise=gradio_params["add_noise"],
        noise_amount=gradio_params["noise_amount"],
        early_stopping_patience=gradio_params["early_stopping_patience"],
        disable_early_stopping=gradio_params["disable_early_stopping"],
        weight_decay=gradio_params["weight_decay"],
        suppress_warnings=gradio_params["suppress_warnings"],
        suppress_reading_messages=gradio_params["suppress_reading_messages"],
        discriminator_update_interval=gradio_params["discriminator_update_interval"],
        label_smoothing_real=gradio_params["label_smoothing_real"],
        label_smoothing_fake=gradio_params["label_smoothing_fake"],
        suppress_detailed_logs=gradio_params["suppress_detailed_logs"],
        use_cache=gradio_params["use_cache"],
        channel_multiplier=gradio_params["channel_multiplier"],
        segments_per_track=gradio_params["segments_per_track"],
        update_cache=gradio_params["update_cache"]
    )

    return 0.0

def start_optuna_optimization(n_trials, gradio_params):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_optuna(trial, gradio_params), n_trials=n_trials)
    return "Optuna optimization completed"

def trial_dirname_creator(trial):
    """Creates a short and unique directory name for each trial."""
    trial_hash = hashlib.sha1(str(trial.number).encode()).hexdigest()
    return f"trial_{trial_hash[:8]}"
