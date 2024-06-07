import os
import torch
import torch.nn as nn
import gradio as gr
from train import start_training_wrapper, stop_training_wrapper, resume_training_wrapper
from separate_stems import perform_separation
from model import load_model
import torchaudio.transforms as T
import logging
import soundfile as sf
import mir_eval
from prepare_dataset import organize_and_prepare_dataset_gradio
from generate_other_noise import generate_shuffled_noise_gradio
from hyperparameter_optimization import objective_optuna, train_ray_tune, start_optuna_optimization, start_ray_tune_optimization
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress httpx, httpcore, urllib3, asyncio, and tensorflow logs below WARNING level
httpx_logger = logging.getLogger("httpx")
httpcore_logger = logging.getLogger("httpcore")
urllib3_logger = logging.getLogger("urllib3")
tensorflow_logger = logging.getLogger("tensorflow")
asyncio_logger = logging.getLogger("asyncio")
httpx_logger.setLevel(logging.WARNING)
httpcore_logger.setLevel(logging.WARNING)
urllib3_logger.setLevel(logging.WARNING)
tensorflow_logger.setLevel(logging.WARNING)
asyncio_logger.setLevel(logging.WARNING)

def read_audio(file_path, suppress_messages=False):
    try:
        if not suppress_messages:
            logger.info(f"Attempting to read: {file_path}")
        data, samplerate = sf.read(file_path)
        return torch.tensor(data).unsqueeze(0), samplerate
    except FileNotFoundError:
        logger.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
    except sf.LibsndfileError as e:
        logger.error(f"Error decoding audio file {file_path}: {e}")
    return None, None

def calculate_metrics(true_audio, predicted_audio, sample_rate):
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(true_audio, predicted_audio)
    return sdr, sir, sar

def perform_separation_wrapper(checkpoint_dir, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    logger.info("Starting separation...")
    result_paths = perform_separation(checkpoint_dir, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages)
    logger.info("Separation completed.")
    return result_paths

def get_checkpoints(checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return []
    return [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

def evaluate_model(input_audio_path, checkpoint_dir, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    input_audio, sr = read_audio(input_audio_path, suppress_messages=suppress_reading_messages)
    if input_audio is None:
        return "Error: Input audio could not be read", "", ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_audio = []

    for stem in range(num_stems):
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_stem_{stem}.pt')
        model = load_model(checkpoint_path, 1, 64, n_mels, target_length, 1, device)  # 1 stem per model
        model.eval()

        with torch.no_grad():
            input_mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(input_audio.float()).unsqueeze(0).to(device)
            output_mel = model(input_mel).cpu()
            inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
            griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32)
            audio = griffin_lim_transform(inverse_mel_transform(output_mel.squeeze(0))).numpy()
            output_audio.append(audio)

    sdr, sir, sar = calculate_metrics(input_audio.numpy(), output_audio, sr)

    return sdr, sir, sar

def log_training_parameters(params):
    logger.info("Training Parameters Selected:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")

with gr.Blocks() as demo:
    with gr.Tab("Training"):
        gr.Markdown("### Train the Model")
        data_dir = gr.Textbox(label="Data Directory", value="K:/KAN-Stem DataSet/ProcessedDataset")
        val_dir = gr.Textbox(label="Validation Directory", value="K:/KAN-Stem DataSet/Chunk_0_Sample")
        batch_size = gr.Number(label="Batch Size", value=16)
        num_epochs = gr.Number(label="Number of Epochs", value=1000)
        learning_rate_g = gr.Number(label="Generator Learning Rate", value=0.03)
        learning_rate_d = gr.Number(label="Discriminator Learning Rate", value=3e-5)
        use_cuda = gr.Checkbox(label="Use CUDA", value=True)
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
        save_interval = gr.Number(label="Save Interval", value=50)
        accumulation_steps = gr.Number(label="Accumulation Steps", value=4)
        num_stems = gr.Number(label="Number of Stems", value=7)
        num_workers = gr.Number(label="Number of Workers", value=1)
        cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        loss_function_g = gr.Dropdown(label="Generator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="L1Loss")
        loss_function_d = gr.Dropdown(label="Discriminator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="WassersteinLoss")
        optimizer_name_g = gr.Dropdown(label="Generator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="SGD")
        optimizer_name_d = gr.Dropdown(label="Discriminator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="RMSProp")
        perceptual_loss_flag = gr.Checkbox(label="Use Perceptual Loss", value=True)
        clip_value = gr.Number(label="Gradient Clipping Value", value=1)
        scheduler_step_size = gr.Number(label="Scheduler Step Size", value=10)
        scheduler_gamma = gr.Number(label="Scheduler Gamma", value=0.9)
        tensorboard_flag = gr.Checkbox(label="Enable TensorBoard Logging", value=True)
        apply_data_augmentation = gr.Checkbox(label="Apply Data Augmentation", value=True)
        add_noise = gr.Checkbox(label="Add Noise", value=True)
        noise_amount = gr.Number(label="Noise Amount", value=0.1)
        early_stopping_patience = gr.Number(label="Early Stopping Patience", value=3)
        disable_early_stopping = gr.Checkbox(label="Disable Early Stopping", value=False)
        weight_decay = gr.Number(label="Weight Decay", value=1e-4)
        suppress_warnings = gr.Checkbox(label="Suppress Warnings", value=True)
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=True)
        use_cpu_for_prep = gr.Checkbox(label="Use CPU for Preparation", value=False)
        discriminator_update_interval = gr.Number(label="Discriminator Update Interval", value=5)
        label_smoothing_real = gr.Slider(label="Label Smoothing Real", minimum=0.7, maximum=0.9, value=0.7, step=0.1)
        label_smoothing_fake = gr.Slider(label="Label Smoothing Fake", minimum=0.1, maximum=0.3, value=0.1, step=0.1)
        perceptual_loss_weight = gr.Number(label="Perceptual Loss Weight", value=0.1)
        suppress_detailed_logs = gr.Checkbox(label="Suppress Detailed Logs", value=True)  # Ensure this is set to True
        use_cache = gr.Checkbox(label="Use Cache", value=True)  # Added use_cache checkbox
        optimization_method = gr.Dropdown(label="Optimization Method", choices=["None", "Optuna", "Ray Tune"], value="Optuna")
        optuna_trials = gr.Number(label="Optuna Trials", value=1)
        ray_samples = gr.Number(label="Ray Tune Samples", value=1)
        start_training_button = gr.Button("Start Training")
        stop_training_button = gr.Button("Stop Training")
        resume_training_button = gr.Button("Resume Training")
        output = gr.Textbox(label="Output")

        def start_training_and_log_params(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                                          accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                                          perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation,
                                          add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep,
                                          discriminator_update_interval, label_smoothing_real, label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs, optimization_method, optuna_trials, ray_samples, use_cache):
            gradio_params = {
                "data_dir": data_dir,
                "val_dir": val_dir,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate_g": learning_rate_g,
                "learning_rate_d": learning_rate_d,
                "use_cuda": use_cuda,
                "checkpoint_dir": checkpoint_dir,
                "save_interval": save_interval,
                "accumulation_steps": accumulation_steps,
                "num_stems": num_stems,
                "num_workers": num_workers,
                "cache_dir": cache_dir,
                "loss_function_g": loss_function_g,
                "loss_function_d": loss_function_d,
                "optimizer_name_g": optimizer_name_g,
                "optimizer_name_d": optimizer_name_d,
                "perceptual_loss_flag": perceptual_loss_flag,
                "clip_value": clip_value,
                "scheduler_step_size": scheduler_step_size,
                "scheduler_gamma": scheduler_gamma,
                "tensorboard_flag": tensorboard_flag,
                "apply_data_augmentation": apply_data_augmentation,
                "add_noise": add_noise,
                "noise_amount": noise_amount,
                "early_stopping_patience": early_stopping_patience,
                "disable_early_stopping": disable_early_stopping,
                "weight_decay": weight_decay,
                "suppress_warnings": suppress_warnings,
                "suppress_reading_messages": suppress_reading_messages,
                "use_cpu_for_prep": use_cpu_for_prep,
                "discriminator_update_interval": discriminator_update_interval,
                "label_smoothing_real": label_smoothing_real,
                "label_smoothing_fake": label_smoothing_fake,
                "perceptual_loss_weight": perceptual_loss_weight,
                "suppress_detailed_logs": suppress_detailed_logs,  # Ensure this key exists
                "use_cache": use_cache  # Ensure this key exists
            }
            log_training_parameters(gradio_params)
            if optimization_method == "Optuna":
                return start_optuna_optimization(optuna_trials, gradio_params)
            elif optimization_method == "Ray Tune":
                return start_ray_tune_optimization(ray_samples, gradio_params)
            else:
                return start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                                          accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                                          perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation,
                                          add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep,
                                          discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, use_cache)  # Added use_cache

        start_training_button.click(
            start_training_and_log_params,
            inputs=[
                data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation,
                add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep,
                discriminator_update_interval, label_smoothing_real, label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs, optimization_method, optuna_trials, ray_samples, use_cache  # Added use_cache
            ],
            outputs=output
        )

        stop_training_button.click(
            stop_training_wrapper,
            outputs=output
        )

        resume_training_button.click(
            resume_training_wrapper,
            inputs=[checkpoint_dir],
            outputs=output
        )

    with gr.Tab("Separation"):
        gr.Markdown("### Perform Separation")
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="path_to_checkpoint_directory")
        file_path = gr.Textbox(label="File Path", value='path_to_input_audio.wav')
        n_mels = gr.Number(label="Number of Mels", value=128)
        target_length = gr.Number(label="Target Length", value=256)
        n_fft = gr.Number(label="Number of FFT", value=2048)
        num_stems = gr.Number(label="Number of Stems", value=7)
        cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.Files(label="Separated Stems")
        perform_separation_button.click(
            perform_separation_wrapper,
            inputs=[checkpoint_dir, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages],
            outputs=result
        )

    with gr.Tab("Evaluation"):
        gr.Markdown("### Evaluate Model")
        eval_checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="path_to_checkpoint_directory")
        eval_file_path = gr.Textbox(label="File Path")
        eval_n_mels = gr.Number(label="Number of Mels", value=128)
        eval_target_length = gr.Number(label="Target Length", value=256)
        eval_n_fft = gr.Number(label="Number of FFT", value=2048)
        eval_num_stems = gr.Number(label="Number of Stems", value=7)
        eval_cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
        eval_button = gr.Button("Evaluate")
        sdr_output = gr.Textbox(label="Signal-to-Distortion Ratio (SDR)")
        sir_output = gr.Textbox(label="Signal-to-Interference Ratio (SIR)")
        sar_output = gr.Textbox(label="Signal-to-Artifacts Ratio (SAR)")
        eval_button.click(
            evaluate_model,
            inputs=[eval_file_path, eval_checkpoint_dir, eval_n_mels, eval_target_length, eval_n_fft, eval_num_stems, eval_cache_dir, suppress_reading_messages],
            outputs=[sdr_output, sir_output, sar_output]
        )

    with gr.Tab("Prepare Dataset"):
        gr.Markdown("### Prepare Dataset")
        input_dir = gr.Textbox(label="Input Directory")
        output_dir = gr.Textbox(label="Output Directory")
        num_examples = gr.Number(label="Number of Examples", value=100)
        prepare_dataset_button = gr.Button("Prepare Dataset")
        prepare_output = gr.Textbox(label="Output")
        prepare_dataset_button.click(
            organize_and_prepare_dataset_gradio,
            inputs=[input_dir, output_dir, num_examples],
            outputs=prepare_output
        )

    with gr.Tab("Generate Other Noise"):
        gr.Markdown("### Generate Shuffled Noise for 'Other' Category")
        noise_input_dir = gr.Textbox(label="Input Directory")
        noise_output_dir = gr.Textbox(label="Output Directory")
        noise_num_examples = gr.Number(label="Number of Examples", value=100)
        generate_noise_button = gr.Button("Generate Noise")
        noise_output = gr.Textbox(label="Output")
        generate_noise_button.click(
            generate_shuffled_noise_gradio,
            inputs=[noise_input_dir, noise_output_dir, noise_num_examples],
            outputs=noise_output
        )

if __name__ == "__main__":
    demo.launch(share=True)
