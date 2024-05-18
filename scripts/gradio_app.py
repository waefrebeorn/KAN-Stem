import gradio as gr
import subprocess
import sys
import os

# Ensure the scripts directory is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from training import train_model
from model import create_model

def start_tensorboard(log_dir):
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

def use_model(audio_file):
    model = create_model()
    # Load the trained model weights (assuming model weights are saved as 'model_weights.h5')
    model.load_weights('model_weights.h5')
    # Process the audio file and make predictions
    audio_data = gr.inputs.Audio()
    predictions = model.predict(audio_data)
    return predictions

def main():
    with gr.Blocks() as demo:
        gr.Markdown('# KAN-Stem Project')
        with gr.Tab('Using the Model'):
            audio_input = gr.Audio(source="microphone", type="filepath", label="Input Audio")
            model_output = gr.Textbox(label='Model Output')
            run_model = gr.Button('Run Model')
            run_model.click(use_model, inputs=audio_input, outputs=model_output)
        with gr.Tab('Training'):
            dataset_path = gr.Textbox(label='Dataset Path')
            start_training = gr.Button('Start Training')
            training_output = gr.Textbox(label='Training Output')
            start_training.click(train_model, inputs=dataset_path, outputs=training_output)
            start_tensorboard_button = gr.Button('Start TensorBoard')
            tensorboard_logdir = gr.Textbox(value='logs/fit', visible=False)
            tensorboard_output = gr.Textbox(label='TensorBoard Output')
            start_tensorboard_button.click(start_tensorboard, inputs=tensorboard_logdir, outputs=tensorboard_output)
    demo.launch()

if __name__ == '__main__':
    main()
