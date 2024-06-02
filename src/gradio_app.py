import os
import gradio as gr
import torch
import torch.nn as nn
from train import start_training_wrapper, stop_training_wrapper
from separate_stems import perform_separation
from model import load_model
import torchaudio.transforms as T
import logging
import soundfile as sf
import mir_eval
from prepare_dataset import organize_and_prepare_dataset_gradio
from generate_other_noise import generate_shuffled_noise_gradio

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s')

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

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    logger.info("Starting separation...")
    result_paths = perform_separation(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages)
    logger.info("Separation completed.")
    return result_paths

def get_checkpoints(checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return []
    return [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

def evaluate_model(input_audio_path, checkpoint_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages):
    input_audio, sr = read_audio(input_audio_path, suppress_messages=suppress_reading_messages)
    if input_audio is None:
        return "Error: Input audio could not be read", "", ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, 1, 64, n_mels, target_length, num_stems, device)
    model.eval()

    with torch.no_grad():
        input_mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(input_audio.float()).unsqueeze(0).to(device)
        output_mel = model(input_mel).cpu()
    
    inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32)
    output_audio = griffin_lim_transform(inverse_mel_transform(output_mel.squeeze(0))).numpy()

    sdr, sir, sar = calculate_metrics(input_audio.numpy(), output_audio, sr)

    return sdr, sir, sar

with gr.Blocks() as demo:
    with gr.Tab("Training"):
        gr.Markdown("### Train the Model")
        data_dir = gr.Textbox(label="Data Directory")
        val_dir = gr.Textbox(label="Validation Directory")
        batch_size = gr.Number(label="Batch Size", value=4)
        num_epochs = gr.Number(label="Number of Epochs", value=10)
        learning_rate_g = gr.Number(label="Generator Learning Rate", value=0.001)
        learning_rate_d = gr.Number(label="Discriminator Learning Rate", value=0.00005)
        use_cuda = gr.Checkbox(label="Use CUDA", value=True)
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
        save_interval = gr.Number(label="Save Interval", value=1)
        accumulation_steps = gr.Number(label="Accumulation Steps", value=4)
        num_stems = gr.Number(label="Number of Stems", value=7)
        num_workers = gr.Number(label="Number of Workers", value=4)
        cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        loss_function_g = gr.Dropdown(label="Generator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="MSELoss")
        loss_function_d = gr.Dropdown(label="Discriminator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="BCEWithLogitsLoss")
        optimizer_name_g = gr.Dropdown(label="Generator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="Adam")
        optimizer_name_d = gr.Dropdown(label="Discriminator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="Adam")
        perceptual_loss_flag = gr.Checkbox(label="Use Perceptual Loss", value=True)
        clip_value = gr.Number(label="Gradient Clipping Value", value=1.0)
        scheduler_step_size = gr.Number(label="Scheduler Step Size", value=5)
        scheduler_gamma = gr.Number(label="Scheduler Gamma", value=0.5)
        tensorboard_flag = gr.Checkbox(label="Enable TensorBoard Logging", value=True)
        apply_data_augmentation = gr.Checkbox(label="Apply Data Augmentation", value=False)
        add_noise = gr.Checkbox(label="Add Noise", value=False)
        noise_amount = gr.Number(label="Noise Amount", value=0.1)
        early_stopping_patience = gr.Number(label="Early Stopping Patience", value=3)
        weight_decay = gr.Number(label="Weight Decay", value=1e-4)
        suppress_warnings = gr.Checkbox(label="Suppress Warnings", value=False)
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
        use_cpu_for_prep = gr.Checkbox(label="Use CPU for Preparation", value=True)
        suppress_detailed_logs = gr.Checkbox(label="Suppress Detailed Logs", value=False)
        start_training_button = gr.Button("Start Training")
        stop_training_button = gr.Button("Stop Training")
        output = gr.Textbox(label="Output")
        start_training_button.click(
            start_training_wrapper,
            inputs=[
                data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation,
                add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings, suppress_reading_messages, use_cpu_for_prep,
                suppress_detailed_logs
            ],
            outputs=output
        )
        stop_training_button.click(
            stop_training_wrapper,
            outputs=output
        )

    with gr.Tab("Separation"):
        gr.Markdown("### Perform Separation")
        checkpoint_path = gr.Dropdown(label="Checkpoint Path", choices=get_checkpoints(), value=None, allow_custom_value=True)
        file_path = gr.Textbox(label="File Path")
        n_mels = gr.Number(label="Number of Mels", value=64)
        target_length = gr.Number(label="Target Length", value=256)
        n_fft = gr.Number(label="Number of FFT", value=1024)
        num_stems = gr.Number(label="Number of Stems", value=7)
        cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.File(label="Separated Stems")
        perform_separation_button.click(
            perform_separation_wrapper,
            inputs=[checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir, suppress_reading_messages],
            outputs=result
        )

    with gr.Tab("Evaluation"):
        gr.Markdown("### Evaluate Model")
        eval_checkpoint_path = gr.Dropdown(label="Checkpoint Path", choices=get_checkpoints(), value=None, allow_custom_value=True)
        eval_file_path = gr.Textbox(label="File Path")
        eval_n_mels = gr.Number(label="Number of Mels", value=64)
        eval_target_length = gr.Number(label="Target Length", value=256)
        eval_n_fft = gr.Number(label="Number of FFT", value=1024)
        eval_num_stems = gr.Number(label="Number of Stems", value=7)
        eval_cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
        eval_button = gr.Button("Evaluate")
        sdr_output = gr.Textbox(label="Signal-to-Distortion Ratio (SDR)")
        sir_output = gr.Textbox(label="Signal-to-Interference Ratio (SIR)")
        sar_output = gr.Textbox(label="Signal-to-Artifacts Ratio (SAR)")
        eval_button.click(
            evaluate_model,
            inputs=[eval_file_path, eval_checkpoint_path, eval_n_mels, eval_target_length, eval_n_fft, eval_num_stems, eval_cache_dir, suppress_reading_messages],
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
