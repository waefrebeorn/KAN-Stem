import os
import logging
import gradio as gr
import torch.nn as nn
from train_single_stem import start_training
from separate_stems import perform_separation
from model import KANWithDepthwiseConv, KANDiscriminator, load_model  # Ensure these imports match your model file location

# Suppress TensorFlow oneDNN custom operations logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress cached_dataset logging
logging.getLogger("cached_dataset").setLevel(logging.WARNING)

# Define the functions to wrap the training and separation logic
def start_training_wrapper(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g_str, loss_function_d_str, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation):
    # Map the selected loss function to the corresponding PyTorch loss function
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }
    loss_function_g = loss_function_map[loss_function_g_str]
    loss_function_d = loss_function_map[loss_function_d_str]
    start_training(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation)
    return f"Training Started with {loss_function_g_str} for Generator and {loss_function_d_str} for Discriminator, and {optimizer_name_g} for Generator Optimizer, {optimizer_name_d} for Discriminator Optimizer"

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir):
    print("Starting separation...")  # Log start of separation
    result_paths = perform_separation(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir)
    print("Separation completed.")  # Log end of separation
    return result_paths

def get_checkpoints(checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return []
    return [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

with gr.Blocks() as demo:
    with gr.Tab("Training"):
        gr.Markdown("### Train the Model")
        data_dir = gr.Textbox(label="Data Directory")
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
        loss_function_g = gr.Dropdown(label="Generator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss"], value="MSELoss")
        loss_function_d = gr.Dropdown(label="Discriminator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss"], value="BCEWithLogitsLoss")
        optimizer_name_g = gr.Dropdown(label="Generator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="Adam")
        optimizer_name_d = gr.Dropdown(label="Discriminator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="Adam")
        perceptual_loss_flag = gr.Checkbox(label="Use Perceptual Loss", value=True)
        clip_value = gr.Number(label="Gradient Clipping Value", value=1.0)
        scheduler_step_size = gr.Number(label="Scheduler Step Size", value=5)
        scheduler_gamma = gr.Number(label="Scheduler Gamma", value=0.5)
        apply_data_augmentation = gr.Checkbox(label="Apply Data Augmentation", value=False)
        start_training_button = gr.Button("Start Training")
        output = gr.Textbox(label="Output")
        start_training_button.click(
            start_training_wrapper,
            inputs=[data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, apply_data_augmentation],
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
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.File(label="Separated Stems")  # Changed to gr.File for handling multiple files
        perform_separation_button.click(
            perform_separation_wrapper,
            inputs=[checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir],
            outputs=result
        )

if __name__ == "__main__":
    demo.launch()
