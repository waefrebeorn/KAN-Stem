import gradio as gr
from train import start_training, test_cuda
from perform_separation import perform_separation  # Import the new function

# Define the functions to wrap the training and separation logic
def start_training_wrapper(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval):
    start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval)
    return "Training Started"

def perform_separation_wrapper(checkpoint_path, file_path, n_mels, target_length):
    result_path = perform_separation(checkpoint_path, file_path, n_mels, target_length)
    return result_path

with gr.Blocks() as demo:
    with gr.Tab("Train Model"):
        gr.Markdown("## CUDA Test and KAN Training")

        btn_cuda = gr.Button("Check CUDA Availability")
        output_cuda = gr.Textbox()
        btn_cuda.click(test_cuda, outputs=output_cuda)

        gr.Markdown("### Start Training")
        data_dir = gr.Textbox(label="Dataset Directory", value="/path/to/your/dataset")
        batch_size = gr.Number(label="Batch Size", value=4)
        num_epochs = gr.Number(label="Number of Epochs", value=10)
        learning_rate = gr.Number(label="Learning Rate", value=0.001)
        use_cuda = gr.Checkbox(label="Use CUDA", value=True)
        checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value="./checkpoints")
        save_interval = gr.Number(label="Checkpoint Save Interval (epochs)", value=1)

        btn_train = gr.Button("Start Training")
        output_train = gr.Textbox()

        btn_train.click(
            start_training_wrapper,
            inputs=[data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval],
            outputs=output_train
        )

    with gr.Tab("Separate Stems"):
        gr.Markdown("## Stem Separation using KAN Model")

        checkpoint_path = gr.Textbox(label="Checkpoint Path", value="./checkpoints/model_epoch_10.pt")
        file_path = gr.Textbox(label="Audio File Path", value="/path/to/your/audio/file.wav")
        n_mels = gr.Number(label="Number of Mel Bands", value=64)
        target_length = gr.Number(label="Target Length", value=256)

        btn_separate = gr.Button("Separate Stems")
        output_separation = gr.Audio(label="Separated Audio")

        btn_separate.click(
            perform_separation_wrapper,
            inputs=[checkpoint_path, file_path, n_mels, target_length],
            outputs=output_separation
        )

# Launch the demo
demo.launch(server_name="127.0.0.1", server_port=7861)
