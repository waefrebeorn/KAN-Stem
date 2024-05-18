import gradio as gr
import subprocess
import sys
import os

# Ensure the scripts directory is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from training import train_model

def start_tensorboard(log_dir):
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

def main():
    with gr.Blocks() as demo:
        gr.Markdown('# KAN-Stem Training')
        with gr.Tab('Training'):
            dataset_path = gr.Textbox(label='Dataset Path')
            start_training = gr.Button('Start Training')
            training_output = gr.Textbox(label='Training Output')
            start_training.click(train_model, inputs=dataset_path, outputs=training_output)
            start_tensorboard_button = gr.Button('Start TensorBoard')
            tensorboard_output = gr.Textbox(label='TensorBoard Output')
            start_tensorboard_button.click(start_tensorboard, inputs='logs/fit', outputs=tensorboard_output)
    demo.launch()

if __name__ == '__main__':
    main()
