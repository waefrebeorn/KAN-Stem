import gradio as gr
import torch
from train import start_training, test_cuda
from separate_stems import perform_separation

# Define the functions to wrap the training and separation logic
def start_training_wrapper(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval):
    start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval)
    return "Training Started"

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length):
    result_paths = perform_separation(checkpoint_path, file_path, n_mels, target_length)
    return result_paths

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
        start_training_button = gr.Button("Start Training")
        output = gr.Textbox(label="Output")
        start_training_button.click(start_training_wrapper, inputs=[data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval], outputs=output)

    with gr.Tab("Separation"):
        gr.Markdown("### Perform Separation")
        checkpoint_path = gr.Textbox(label="Checkpoint Path")
        file_path = gr.Textbox(label="File Path")
        n_mels = gr.Number(label="Number of Mels", value=64)
        target_length = gr.Number(label="Target Length", value=256)
        perform_separation_button = gr.Button("Perform Separation")
        result = gr.File(label="Separated Stems")
        perform_separation_button.click(perform_separation_wrapper, inputs=[checkpoint_path, file_path, n_mels, target_length], outputs=result)

if __name__ == "__main__":
    demo.launch()
