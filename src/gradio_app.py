import os
import logging
import gradio as gr
import torch
import torch.nn as nn
from multiprocessing import Process
from train_single_stem import start_training
from separate_stems import perform_separation
from model import KANWithDepthwiseConv, load_model
import soundfile as sf
import torchaudio.transforms as T
import mir_eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("cached_dataset").setLevel(logging.WARNING)

training_process = None

def read_audio(file_path):
    try:
        print(f"Attempting to read: {file_path}")
        data, samplerate = sf.read(file_path)
        return torch.tensor(data).unsqueeze(0), samplerate
    except FileNotFoundError:
        logging.error(f"Error: Audio file not found: {file_path}")
    except RuntimeError as e:
        logging.error(f"Error reading audio file {file_path}: {e}")
    except sf.LibsndfileError as e:
        logging.error(f"Error decoding audio file {file_path}: {e}")
    return None, None

def calculate_metrics(true_audio, predicted_audio, sample_rate):
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(true_audio, predicted_audio)
    return sdr, sir, sar

def run_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings):
    start_training(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings)

def start_training_wrapper(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings):
    global training_process
    loss_function_map = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }
    loss_function_g = loss_function_map[loss_function_str_g]
    loss_function_d = loss_function_map[loss_function_str_d]
    training_process = Process(target=run_training, args=(data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings))
    training_process.start()
    return f"Training Started with {loss_function_str_g} for Generator and {loss_function_str_d} for Discriminator, using {optimizer_name_g} for Generator Optimizer and {optimizer_name_d} for Discriminator Optimizer"

def stop_training_wrapper():
    global training_process
    if training_process is not None:
        training_process.terminate()
        training_process = None
        return "Training Stopped"
    return "No Training Process Running"

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir):
    print("Starting separation...")
    result_paths = perform_separation(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir)
    print("Separation completed.")
    return result_paths

def get_checkpoints(checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return []
    return [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

def evaluate_model(input_audio_path, checkpoint_path, n_mels, target_length, n_fft, num_stems, cache_dir):
    # Load the input audio
    input_audio, sr = read_audio(input_audio_path)
    if input_audio is None:
        return "Error: Input audio could not be read", "", ""

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, 1, 64, n_mels, target_length, num_stems, device)
    model.eval()

    # Perform inference
    with torch.no_grad():
        input_mel = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(input_audio.float()).unsqueeze(0).to(device)
        output_mel = model(input_mel).cpu()
    
    # Inverse mel-spectrogram to get the audio back
    inverse_mel_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
    griffin_lim_transform = T.GriffinLim(n_fft=n_fft, n_iter=32)
    output_audio = griffin_lim_transform(inverse_mel_transform(output_mel.squeeze(0))).numpy()

    # Calculate metrics
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
        loss_function_g = gr.Dropdown(label="Generator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss"], value="MSELoss")
        loss_function_d = gr.Dropdown(label="Discriminator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss"], value="BCEWithLogitsLoss")
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
        start_training_button = gr.Button("Start Training")
        stop_training_button = gr.Button("Stop Training")
        output = gr.Textbox(label="Output")
        start_training_button.click(
            start_training_wrapper,
            inputs=[data_dir, val_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d, perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, apply_data_augmentation, add_noise, noise_amount, early_stopping_patience, weight_decay, suppress_warnings],
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
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.File(label="Separated Stems")
        perform_separation_button.click(
            perform_separation_wrapper,
            inputs=[checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems, cache_dir],
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
        eval_button = gr.Button("Evaluate")
        sdr_output = gr.Textbox(label="Signal-to-Distortion Ratio (SDR)")
        sir_output = gr.Textbox(label="Signal-to-Interference Ratio (SIR)")
        sar_output = gr.Textbox(label="Signal-to-Artifacts Ratio (SAR)")
        eval_button.click(
            evaluate_model,
            inputs=[eval_file_path, eval_checkpoint_path, eval_n_mels, eval_target_length, eval_n_fft, eval_num_stems, eval_cache_dir],
            outputs=[sdr_output, sir_output, sar_output]
        )

if __name__ == "__main__":
    demo.launch()
