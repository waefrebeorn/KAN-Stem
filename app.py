import os
import gradio as gr
import torch
import librosa
import numpy as np

# import requests  # Commenting out since it's not used

from modules.KANModel import KANModel
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Function to load data from a given directory
def load_stem_data(dataset_path):
    try:
        wav_path = os.path.join(dataset_path, 'wav')
        wav_files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]
        return wav_files
    except Exception as e:
        print(f"Error loading stem data: {e}")
        return []

# Preprocess function
def preprocess(audio, max_duration):
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    elif audio.ndim != 2:
        raise ValueError("Audio data must be at least one-dimensional and not empty")

    num_samples = int(max_duration * 44100)
    if audio.shape[1] > num_samples:
        audio = audio[:, :num_samples]
    elif audio.shape[1] < num_samples:
        padding = np.zeros((audio.shape[0], num_samples - audio.shape[1]))
        audio = np.hstack((audio, padding))

    return torch.from_numpy(audio).float()

# Load the model
def load_model(checkpoint_path):
    model = KANModel()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Separate audio
def separate_audio(input_audio, max_duration, model):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_data = preprocess(input_audio, max_duration).to(device)
        with torch.no_grad():
            separated_data = model(input_data)
        return separated_data.cpu().numpy()
    except Exception as e:
        print(f"Error in separate_audio: {e}")
        return []

# Define the Gradio interface
def gradio_interface():
    def separate(input_audio):
        model = load_model("checkpoints/model.ckpt")
        max_duration = 10.0  # 10 seconds
        separated_audio = separate_audio(input_audio, max_duration, model)
        return separated_audio

    iface = gr.Interface(
        fn=separate,
        inputs=gr.inputs.Audio(source="microphone", type="numpy"),
        outputs=gr.outputs.Audio(type="numpy"),
        live=True,
    )
    return iface

# Launch the app
if __name__ == "__main__":
    iface = gradio_interface()
    iface.launch(server_name="127.0.0.1")
