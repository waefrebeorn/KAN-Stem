import os
import torch
import torch.nn as nn
import gradio as gr
from multiprocessing import Value, Process, Manager
from train import start_training_wrapper, stop_training_wrapper, save_checkpoint_gradio, resume_training, start_training
from separate_stems import perform_separation
from model import load_model
import torchaudio.transforms as T
import logging
import soundfile as sf
import mir_eval
from prepare_dataset import organize_and_prepare_dataset_gradio
from generate_other_noise import generate_shuffled_noise_gradio
from hyperparameter_optimization import objective_optuna, start_optuna_optimization
from parse_event_file import parse_event_file
from model_setup import create_model_and_optimizer, initialize_model
import warnings
import tensorflow as tf
import time

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")
warnings.filterwarnings("ignore", message="unable to parse version details from package URL.", module="gradio.analytics")

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
        model = load_model(checkpoint_path, 3, 64, n_mels, target_length, 1, device)
        model.eval()

        with torch.no_grad():
            mel_spectrogram = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(input_audio.float()).unsqueeze(0).to(device)
            harmonic, percussive = librosa.decompose.hpss(mel_spectrogram.cpu().numpy())
            harmonic_t = torch.from_numpy(harmonic).to(device)
            percussive_t = torch.from_numpy(percussive).to(device)
            input_mel = torch.stack([mel_spectrogram, harmonic_t, percussive_t], dim=-1)

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

def stop_training():
    global stop_flag
    stop_flag.value = 1
    return stop_training_wrapper(stop_flag)

def start_training_and_log_params(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                                  accumulation_steps, num_stems, num_workers, cache_dir, segments_per_track, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                                  perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, 
                                  add_noise, noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
                                  discriminator_update_interval, label_smoothing_real, label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs,
                                  optimization_method, optuna_trials, use_cache, channel_multiplier, update_cache, selected_stems):
    global training_state
    gradio_params = {
        "data_dir": data_dir,
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
        "segments_per_track": segments_per_track,  
        "loss_function_g": loss_function_g,
        "loss_function_d": loss_function_d,
        "optimizer_name_g": optimizer_name_g,
        "optimizer_name_d": optimizer_name_d,
        "perceptual_loss_flag": perceptual_loss_flag,
        "clip_value": clip_value,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "tensorboard_flag": tensorboard_flag,
        "add_noise": add_noise,
        "noise_amount": noise_amount,
        "early_stopping_patience": early_stopping_patience,
        "disable_early_stopping": disable_early_stopping,
        "weight_decay": weight_decay,
        "suppress_warnings": suppress_warnings,
        "suppress_reading_messages": suppress_reading_messages,
        "discriminator_update_interval": discriminator_update_interval,
        "label_smoothing_real": label_smoothing_real,
        "label_smoothing_fake": label_smoothing_fake,
        "perceptual_loss_weight": perceptual_loss_weight,
        "suppress_detailed_logs": suppress_detailed_logs,
        "use_cache": use_cache,
        "channel_multiplier": channel_multiplier,  
        "update_cache": update_cache,
        "selected_stems": selected_stems
    }
    log_training_parameters(gradio_params)
    training_state.update({
        "model_params": gradio_params,
        "training_started": True
    })

    if optimization_method == "Optuna":
        return start_optuna_optimization(optuna_trials, gradio_params)
    else:
        return start_training_wrapper(data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                                      accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                                      perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise,
                                      noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
                                      discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs,
                                      use_cache, channel_multiplier, segments_per_track, update_cache, training_state, stop_flag, checkpoint_flag, selected_stems)

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, real_output, fake_output):
        return torch.mean(fake_output) - torch.mean(real_output)

loss_functions = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "WassersteinLoss": WassersteinLoss  # assuming WassersteinLoss is defined elsewhere
}

def resume_training_wrapper(
    selected_checkpoint, checkpoint_dir, data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, save_interval,
    accumulation_steps, num_stems, num_workers, cache_dir, segments_per_track, loss_function_str_g, loss_function_str_d, optimizer_name_g, optimizer_name_d,
    perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise, noise_amount, early_stopping_patience,
    disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, discriminator_update_interval, label_smoothing_real,
    label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs, use_cache, channel_multiplier, update_cache, selected_stems
):
    try:
        logger.info(f"Loading selected checkpoint: {selected_checkpoint}")
        checkpoint = torch.load(selected_checkpoint, map_location='cpu')

        logger.info("Checkpoint loaded successfully.")

        training_params = {
            'data_dir': data_dir,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate_g': learning_rate_g,
            'learning_rate_d': learning_rate_d,
            'use_cuda': use_cuda,
            'save_interval': save_interval,
            'accumulation_steps': accumulation_steps,
            'num_stems': num_stems,
            'num_workers': num_workers,
            'cache_dir': cache_dir,
            'segments_per_track': segments_per_track,
            'loss_function_str_g': loss_function_str_g,
            'loss_function_str_d': loss_function_str_d,
            'optimizer_name_g': optimizer_name_g,
            'optimizer_name_d': optimizer_name_d,
            'perceptual_loss_flag': perceptual_loss_flag,
            'clip_value': clip_value,
            'scheduler_step_size': scheduler_step_size,
            'scheduler_gamma': scheduler_gamma,
            'tensorboard_flag': tensorboard_flag,
            'add_noise': add_noise,
            'noise_amount': noise_amount,
            'early_stopping_patience': early_stopping_patience,
            'disable_early_stopping': disable_early_stopping,
            'weight_decay': weight_decay,
            'suppress_warnings': suppress_warnings,
            'suppress_reading_messages': suppress_reading_messages,
            'discriminator_update_interval': discriminator_update_interval,
            'label_smoothing_real': label_smoothing_real,
            'label_smoothing_fake': label_smoothing_fake,
            'perceptual_loss_weight': perceptual_loss_weight,
            'suppress_detailed_logs': suppress_detailed_logs,
            'use_cache': use_cache,
            'channel_multiplier': channel_multiplier,
            'update_cache': update_cache,
            'selected_stems': selected_stems
        }

        logger.info("Creating model and optimizer.")
        device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        target_length = checkpoint.get('target_length', checkpoint.get('segment_length', 22050))
        model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = create_model_and_optimizer(
            device=device,
            n_mels=checkpoint['n_mels'],
            target_length=target_length,
            initial_lr_g=training_params['learning_rate_g'],
            initial_lr_d=training_params['learning_rate_d'],
            optimizer_name_g=training_params['optimizer_name_g'],
            optimizer_name_d=training_params['optimizer_name_d'],
            weight_decay=training_params['weight_decay']
        )

        logger.info("Loading state dictionaries.")
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
        scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])

        model.to(device)
        discriminator.to(device)

        training_state = {
            'model': model,
            'discriminator': discriminator,
            'optimizer_g': optimizer_g,
            'optimizer_d': optimizer_d,
            'scaler_g': scaler_g,
            'scaler_d': scaler_d,
            'model_params': checkpoint.get('model_params', {}),
            'training_params': training_params,
            'stem_name': checkpoint.get('stem_name'),
            'current_epoch': checkpoint.get('epoch', 0),
            'current_segment': checkpoint.get('segment', 0),
            'training_started': True,
            'target_length': target_length
        }

        logger.info("Mapping loss functions from strings to actual functions.")
        loss_function_map = {
            "MSELoss": nn.MSELoss,
            "L1Loss": nn.L1Loss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "WassersteinLoss": WassersteinLoss
        }
        loss_function_g = loss_function_map[loss_function_str_g]()
        loss_function_d = loss_function_map[loss_function_str_d]()

        logger.info("Resuming training.")
        training_process = Process(target=start_training, args=(
            data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
            accumulation_steps, num_stems, num_workers, cache_dir, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
            perceptual_loss_flag, perceptual_loss_weight, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise,
            noise_amount, early_stopping_patience, disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages,
            discriminator_update_interval, label_smoothing_real, label_smoothing_fake, suppress_detailed_logs, stop_flag, checkpoint_flag,
            training_state, use_cache, channel_multiplier, segments_per_track, update_cache, selected_stems,  # Pass selected_stems
            training_state['current_segment']  # Pass current_segment
        ))
        training_process.start()

        return f"Resumed training from checkpoint: {selected_checkpoint}"
    except KeyError as e:
        logger.error(f"KeyError during checkpoint loading: {e}")
        return f"KeyError during checkpoint loading: {e}"
    except TypeError as e:
        logger.error(f"TypeError during checkpoint loading: {e}")
        return f"TypeError during checkpoint loading: {e}"
    except Exception as e:
        logger.error(f"Exception during checkpoint loading: {e}", exc_info=True)
        return f"Exception during checkpoint loading: {e}"

def update_checkpoint_dropdown(checkpoint_dir):
    checkpoints = get_checkpoints(checkpoint_dir)
    return gr.update(choices=checkpoints)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # For frozen executables

    # Define global variables (Move them inside __main__)
    stop_flag = Value('i', 0)
    checkpoint_flag = Value('i', 0)
    training_process = None

    # Global training state dictionary (use a Manager to make it shareable)
    manager = Manager()
    training_state = manager.dict({
        # Initialize your training state here
        "model": None,
        "optimizer_g": None,
        "optimizer_d": None,
        "scaler_g": None,
        "scaler_d": None,
        "model_params": {},
        "training_params": {},
        "stem_name": None,
        "current_epoch": 0,
        "current_segment": 0,
        "training_started": False
    })

    # Ensure this block is guarded to prevent recursive spawning issues on Windows
    multiprocessing.set_start_method("spawn", force=True)

    with gr.Blocks() as demo:
        with gr.Tab("Training"):
            gr.Markdown("### Train the Model")
            data_dir = gr.Textbox(label="Data Directory", value="K:/KAN-Stem DataSet/prepared dataset")
            batch_size = gr.Number(label="Batch Size", value=1)
            num_epochs = gr.Number(label="Number of Epochs", value=10)
            learning_rate_g = gr.Number(label="Generator Learning Rate", value=0.03)
            learning_rate_d = gr.Number(label="Discriminator Learning Rate", value=3e-5)
            use_cuda = gr.Checkbox(label="Use CUDA", value=True)
            checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
            save_interval = gr.Number(label="Save Interval", value=1)
            accumulation_steps = gr.Number(label="Accumulation Steps", value=1)
            num_stems = gr.Number(label="Number of Stems", value=6)
            num_workers = gr.Number(label="Number of Workers", value=1)
            cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
            segments_per_track = gr.Number(label="Segments per Track", value=1)
            loss_function_g = gr.Dropdown(label="Generator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="L1Loss")
            loss_function_d = gr.Dropdown(label="Discriminator Loss Function", choices=["MSELoss", "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss", "WassersteinLoss"], value="WassersteinLoss")
            optimizer_name_g = gr.Dropdown(label="Generator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="SGD")
            optimizer_name_d = gr.Dropdown(label="Discriminator Optimizer", choices=["SGD", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adam"], value="RMSProp")
            perceptual_loss_flag = gr.Checkbox(label="Use Perceptual Loss", value=False)
            clip_value = gr.Number(label="Gradient Clipping Value", value=1)
            scheduler_step_size = gr.Number(label="Scheduler Step Size", value=10)
            scheduler_gamma = gr.Number(label="Scheduler Gamma", value=0.9)
            tensorboard_flag = gr.Checkbox(label="Enable TensorBoard Logging", value=True)
            add_noise = gr.Checkbox(label="Add Noise", value=True)
            noise_amount = gr.Number(label="Noise Amount", value=0.1)
            early_stopping_patience = gr.Number(label="Early Stopping Patience", value=25)
            disable_early_stopping = gr.Checkbox(label="Disable Early Stopping", value=False)
            weight_decay = gr.Number(label="Weight Decay", value=1e-4)
            suppress_warnings = gr.Checkbox(label="Suppress Warnings", value=False)
            suppress_reading_messages = gr.Checkbox(label="Suppress Reading Messages", value=False)
            discriminator_update_interval = gr.Number(label="Discriminator Update Interval", value=1)
            label_smoothing_real = gr.Slider(label="Label Smoothing Real", minimum=0.7, maximum=0.9, value=0.7, step=0.1)
            label_smoothing_fake = gr.Slider(label="Label Smoothing Fake", minimum=0.1, maximum=0.3, value=0.1, step=0.1)
            perceptual_loss_weight = gr.Number(label="Perceptual Loss Weight", value=0.1)
            suppress_detailed_logs = gr.Checkbox(label="Suppress Detailed Logs", value=False)
            use_cache = gr.Checkbox(label="Use Cache", value=True)
            optimization_method = gr.Dropdown(label="Optimization Method", choices=["None", "Optuna"], value="None")
            optuna_trials = gr.Number(label="Optuna Trials", value=1)
            channel_multiplier = gr.Number(label="Channel Multiplier", value=0.5)
            update_cache = gr.Checkbox(label="Update Cache", value=True)
            start_training_button = gr.Button("Start Training")
            stop_training_button = gr.Button("Stop Training")
            resume_checkpoint_dropdown = gr.Dropdown(label="Select Checkpoint", choices=get_checkpoints(), value=None)
            refresh_checkpoint_button = gr.Button("Refresh Checkpoints")
            resume_training_button = gr.Button("Resume Training")
            save_checkpoint_button = gr.Button("Save Checkpoint")
            output = gr.Textbox(label="Output")

            # Stem selection
            stem_selector = gr.CheckboxGroup(
                label="Select Stems to Train",
                choices=["vocals", "drums", "bass", "kick", "keys", "guitar"],
                value=["vocals"]  # Default value
            )

            start_training_button.click(
                start_training_and_log_params,
                inputs=[
                    data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, checkpoint_dir, save_interval,
                    accumulation_steps, num_stems, num_workers, cache_dir, segments_per_track, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                    perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise, noise_amount, early_stopping_patience,
                    disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, discriminator_update_interval, label_smoothing_real, 
                    label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs, optimization_method, optuna_trials, use_cache, channel_multiplier, update_cache,
                    stem_selector
                ],
                outputs=output
            )

            stop_training_button.click(
                lambda: stop_training_wrapper(stop_flag),
                inputs=[],
                outputs=output
            )

            refresh_checkpoint_button.click(
                lambda chkpt_dir: update_checkpoint_dropdown(chkpt_dir),
                inputs=[checkpoint_dir],
                outputs=resume_checkpoint_dropdown
            )

            resume_training_button.click(
                resume_training_wrapper,
                inputs=[
                    resume_checkpoint_dropdown, checkpoint_dir, data_dir, batch_size, num_epochs, learning_rate_g, learning_rate_d, use_cuda, save_interval,
                    accumulation_steps, num_stems, num_workers, cache_dir, segments_per_track, loss_function_g, loss_function_d, optimizer_name_g, optimizer_name_d,
                    perceptual_loss_flag, clip_value, scheduler_step_size, scheduler_gamma, tensorboard_flag, add_noise, noise_amount, early_stopping_patience,
                    disable_early_stopping, weight_decay, suppress_warnings, suppress_reading_messages, discriminator_update_interval, label_smoothing_real,
                    label_smoothing_fake, perceptual_loss_weight, suppress_detailed_logs, use_cache, channel_multiplier, update_cache, stem_selector
                ],
                outputs=output
            )
            
            save_checkpoint_button.click(
                lambda chkpt_dir: save_checkpoint_gradio(chkpt_dir, training_state, checkpoint_flag),
                inputs=[checkpoint_dir],
                outputs=output
            )

        with gr.Tab("Separation"):
            gr.Markdown("### Perform Separation")
            checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
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
            eval_checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
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
            target_length = gr.Number(label="Target Length (seconds)", value=60)
            sample_rate = gr.Number(label="Sample Rate", value=44100)
            prepare_dataset_button = gr.Button("Prepare Dataset")
            prepare_output = gr.Textbox(label="Output")
            prepare_dataset_button.click(
                organize_and_prepare_dataset_gradio,
                inputs=[input_dir, output_dir, num_examples, target_length, sample_rate],
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

        with gr.Tab("Parse TensorFlow Event File"):
            gr.Markdown("### Parse TensorFlow Event File")
            event_file = gr.Textbox(label="Event File Path")
            output_file = gr.Textbox(label="Output File Path", value="parsed_output.txt")
            max_entries_per_tag = gr.Number(label="Max Entries per Tag", value=120)
            parse_event_button = gr.Button("Parse Event File")
            parsed_output = gr.Textbox(label="Parsed Output")
            
            def parse_and_display_event_file(event_file, output_file, max_entries_per_tag):
                parse_event_file(event_file, output_file, max_entries_per_tag)
                with open(output_file, 'r') as file:
                    return file.read()

            parse_event_button.click(
                parse_and_display_event_file,
                inputs=[event_file, output_file, max_entries_per_tag],
                outputs=parsed_output
            )

    demo.launch(share=False)
