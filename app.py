import gradio as gr
from train import train_model

def start_training(dataset_path, num_epochs, batch_size, learning_rate):
    if not hasattr(start_training, 'running'):
        start_training.running = False

    if start_training.running:
        return 'Training already in progress...'

    start_training.running = True
    train_model(dataset_path, num_epochs, batch_size, learning_rate)
    start_training.running = False
    return 'Training completed.'

def visualize_logs(log_dir):
    # Visualization logic for TensorBoard logs (this can be improved with actual TensorBoard integration)
    return 'Logs visualized.'

with gr.Blocks() as demo:
    gr.Markdown('# KAN-Stem Training App')
    with gr.Tab('Training'):
        dataset_path = gr.Textbox(label='Dataset Path', value='G:\\Music\\badmultitracks-michaeljackson\\dataset')
        num_epochs = gr.Slider(label='Number of Epochs', minimum=1, maximum=100, value=10, step=1)
        batch_size = gr.Slider(label='Batch Size', minimum=1, maximum=64, value=16, step=1)
        learning_rate = gr.Number(label='Learning Rate', value=0.001)
        train_button = gr.Button('Start Training')
        output_text = gr.Textbox()
        train_button.click(start_training, [dataset_path, num_epochs, batch_size, learning_rate], outputs=output_text)

    with gr.Tab('Logs'):
        log_dir = gr.Textbox(label='Log Directory', value='runs')
        visualize_button = gr.Button('Visualize Logs')
        log_output = gr.Textbox()
        visualize_button.click(visualize_logs, log_dir, outputs=log_output)

demo.launch(share=True)
