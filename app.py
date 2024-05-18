import gradio as gr
from train import train_model

def start_training(dataset_path, num_epochs, batch_size, learning_rate):
    train_model(dataset_path, num_epochs, batch_size, learning_rate)
    return 'Training completed.'

def visualize_logs(log_dir):
    # Visualization logic for TensorBoard logs (this can be improved with actual TensorBoard integration)
    return 'Logs visualized.'

with gr.Blocks() as demo:
    gr.Markdown('# KAN-Stem Training App')
    with gr.Tab('Training'):
        dataset_path = gr.Textbox(label='Dataset Path', value='G:\\Music\\badmultitracks-michaeljackson\\dataset')
        num_epochs = gr.Slider(label='Number of Epochs', minimum=1, maximum=100, value=10, step=1)
        batch_size = gr.Slider(label='Batch Size', minimum=1, maximum=128, value=32, step=1)
        learning_rate = gr.Number(label='Learning Rate', value=0.001)
        train_button = gr.Button('Start Training')
        train_button.click(start_training, [dataset_path, num_epochs, batch_size, learning_rate], outputs='text')

    with gr.Tab('Logs'):
        log_dir = gr.Textbox(label='Log Directory', value='runs')
        visualize_button = gr.Button('Visualize Logs')
        visualize_button.click(visualize_logs, log_dir, outputs='text')

demo.launch()
