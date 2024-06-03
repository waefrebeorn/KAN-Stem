import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from train import start_training
import torch.nn as nn
from loss_functions import wasserstein_loss

def objective_optuna(trial):
    batch_size = trial.suggest_int('batch_size', 16, 64)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    learning_rate_g = trial.suggest_loguniform('learning_rate_g', 1e-5, 1e-1)
    learning_rate_d = trial.suggest_loguniform('learning_rate_d', 1e-5, 1e-1)
    perceptual_loss_weight = trial.suggest_uniform('perceptual_loss_weight', 0.0, 1.0)
    clip_value = trial.suggest_uniform('clip_value', 0.5, 1.5)

    # Call the start_training function with the suggested hyperparameters
    start_training(data_dir="data_dir", val_dir="val_dir", batch_size=batch_size, num_epochs=num_epochs,
                   initial_lr_g=learning_rate_g, initial_lr_d=learning_rate_d, use_cuda=True,
                   checkpoint_dir="checkpoint_dir", save_interval=50, accumulation_steps=4, num_stems=7,
                   num_workers=4, cache_dir="cache_dir", loss_function_g=nn.L1Loss(), loss_function_d=wasserstein_loss,
                   optimizer_name_g="Adam", optimizer_name_d="Adam", perceptual_loss_flag=True,
                   perceptual_loss_weight=perceptual_loss_weight, clip_value=clip_value, scheduler_step_size=10,
                   scheduler_gamma=0.9, tensorboard_flag=False, apply_data_augmentation=True, add_noise=True,
                   noise_amount=0.1, early_stopping_patience=3, disable_early_stopping=False, weight_decay=1e-4,
                   suppress_warnings=True, suppress_reading_messages=True, use_cpu_for_prep=False,
                   discriminator_update_interval=5, label_smoothing_real=0.8, label_smoothing_fake=0.2,
                   suppress_detailed_logs=True)

    return 0.0

def train_ray_tune(config):
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate_g = config['learning_rate_g']
    learning_rate_d = config['learning_rate_d']
    perceptual_loss_weight = config['perceptual_loss_weight']
    clip_value = config['clip_value']

    start_training(data_dir="data_dir", val_dir="val_dir", batch_size=batch_size, num_epochs=num_epochs,
                   initial_lr_g=learning_rate_g, initial_lr_d=learning_rate_d, use_cuda=True,
                   checkpoint_dir="checkpoint_dir", save_interval=50, accumulation_steps=4, num_stems=7,
                   num_workers=4, cache_dir="cache_dir", loss_function_g=nn.L1Loss(), loss_function_d=wasserstein_loss,
                   optimizer_name_g="Adam", optimizer_name_d="Adam", perceptual_loss_flag=True,
                   perceptual_loss_weight=perceptual_loss_weight, clip_value=clip_value, scheduler_step_size=10,
                   scheduler_gamma=0.9, tensorboard_flag=False, apply_data_augmentation=True, add_noise=True,
                   noise_amount=0.1, early_stopping_patience=3, disable_early_stopping=False, weight_decay=1e-4,
                   suppress_warnings=True, suppress_reading_messages=True, use_cpu_for_prep=False,
                   discriminator_update_interval=5, label_smoothing_real=0.8, label_smoothing_fake=0.2,
                   suppress_detailed_logs=True)

    tune.report(metric=0.0)

def start_optuna_optimization(n_trials):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_optuna, n_trials=n_trials)
    return "Optuna optimization completed"

def trial_dirname_creator(trial):
    """Creates a short and unique directory name for each trial."""
    trial_hash = hashlib.sha1(trial.trial_id.encode()).hexdigest()
    return f"trial_{trial_hash[:8]}"

def start_ray_tune_optimization(num_samples):
    ray.init()
    config = {
        'batch_size': tune.choice([16, 32, 64]),
        'num_epochs': tune.choice([10, 20, 50, 100]),
        'learning_rate_g': tune.loguniform(1e-5, 1e-1),
        'learning_rate_d': tune.loguniform(1e-5, 1e-1),
        'perceptual_loss_weight': tune.uniform(0.0, 1.0),
        'clip_value': tune.uniform(0.5, 1.5)
    }
    scheduler = ASHAScheduler(
        metric='metric',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    tune.run(train_ray_tune, config=config, scheduler=scheduler, num_samples=num_samples, trial_dirname_creator=trial_dirname_creator)
    return "Ray Tune optimization completed"
