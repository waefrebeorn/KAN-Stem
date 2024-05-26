import os
import logging
import gradio as gr
from train_single_stem import start_training
from separate_stems import perform_separation

# Suppress TensorFlow oneDNN custom operations logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress cached_dataset logging
logging.getLogger("cached_dataset").setLevel(logging.WARNING)

# Define the functions to wrap the training and separation logic
def start_training_wrapper(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir):
    start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir)
    return "Training Started"

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems):
    print("Starting separation...")  # Log start of separation
    result_paths = perform_separation(checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems)
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
        learning_rate = gr.Number(label="Learning Rate", value=0.001)
        use_cuda = gr.Checkbox(label="Use CUDA", value=True)
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
        save_interval = gr.Number(label="Save Interval", value=1)
        accumulation_steps = gr.Number(label="Accumulation Steps", value=4)
        num_stems = gr.Number(label="Number of Stems", value=7)
        num_workers = gr.Number(label="Number of Workers", value=4)
        cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
        start_training_button = gr.Button("Start Training")
        output = gr.Textbox(label="Output")
        start_training_button.click(
            start_training_wrapper,
            inputs=[data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir],
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
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.File(label="Separated Stems")
        perform_separation_button.click(
            perform_separation_wrapper,
            inputs=[checkpoint_path, file_path, n_mels, target_length, n_fft, num_stems],
            outputs=result
        )

if __name__ == "__main__":
    demo.launch()
