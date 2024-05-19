import os
import gradio as gr
import torch
import librosa
import numpy as np
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

# Function to train the model
def train_model(dataset_path, checkpoint_path, log_dir, num_epochs=10):
    model = KANModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        wav_files = load_stem_data(dataset_path)
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(wav_file, sr=None)
                audio = preprocess(audio, max_duration=10.0).to(device)

                optimizer.zero_grad()
                outputs = model(audio)
                loss = criterion(outputs, audio)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

        avg_loss = epoch_loss / len(wav_files)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()

# Define the Gradio interface for training
def gradio_training_interface():
    def train(dataset_path, checkpoint_path, log_dir, num_epochs):
        train_model(dataset_path, checkpoint_path, log_dir, num_epochs)
        return f"Training completed. Model saved to {checkpoint_path}"

    iface = gr.Interface(
        fn=train,
        inputs=[
            gr.Textbox(label="Dataset Path"),
            gr.Textbox(label="Checkpoint Path", value="checkpoints/model.ckpt"),
            gr.Textbox(label="Log Directory", value="logs"),
            gr.Slider(label="Number of Epochs", minimum=1, maximum=100, value=10)
        ],
        outputs="text"
    )
    return iface

# Define the Gradio interface for inference
def gradio_inference_interface():
    def separate(input_audio):
        model = load_model("checkpoints/model.ckpt")
        max_duration = 10.0  # 10 seconds
        separated_audio = separate_audio(input_audio, max_duration, model)
        return separated_audio

    iface = gr.Interface(
        fn=separate,
        inputs=gr.Audio(type="numpy"),
        outputs=gr.Audio(type="numpy"),
        live=True,
    )
    return iface

# Launch the app
if __name__ == "__main__":
    train_iface = gradio_training_interface()
    infer_iface = gradio_inference_interface()

    app = gr.TabbedInterface([train_iface, infer_iface], ["Train Model", "Separate Audio"])
    app.launch(server_name="127.0.0.1")
