import os
import gradio as gr
import torch
import librosa
import numpy as np
from modules.KANModel import KANModel  # Ensure correct import
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Function to load data from a given directory
def load_stem_data(dataset_path):
    wav_path = os.path.join(dataset_path, 'wav')
    wav_files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]

    inputs = []
    targets = []

    for wav_file in wav_files:
        y, sr = librosa.load(wav_file, sr=None)
        y = y.astype(np.float32)  # Ensure audio data is floating-point
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Ensure the log_spectrogram has the correct shape
        target_shape = (128, 128 * 128)
        if log_spectrogram.shape[1] < target_shape[1]:
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, target_shape[1] - log_spectrogram.shape[1])), 'constant')
        else:
            log_spectrogram = log_spectrogram[:, :target_shape[1]]

        inputs.append(log_spectrogram)

        # Example targets, in practice, load actual target data
        target = np.random.randn(4, 44100)  # Example target with 4 stems
        targets.append(target)

    inputs = np.array(inputs)
    targets = np.array(targets)

    # Ensure inputs have the correct shape
    inputs = inputs[:, np.newaxis, :, :]  # Add a channel dimension

    return inputs, targets

# Training function
def train_model(epochs, learning_rate, batch_size, dataset_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs, targets = load_stem_data(dataset_path)
    model = KANModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # TensorBoard writer
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(inputs), batch_size):
            input_batch = inputs[i:i + batch_size]
            target_batch = targets[i:i + batch_size]
            input_batch = torch.tensor(input_batch, dtype=torch.float32).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (len(inputs) // batch_size)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
        writer.add_scalar('Loss/train', avg_loss, epoch)

    writer.close()
    torch.save(model.state_dict(), os.path.join('checkpoints', 'model.ckpt'))
    return 'Model trained and saved at checkpoints/model.ckpt'

# Function to get a list of model checkpoints
def get_model_checkpoints(checkpoint_path):
    return [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]

# Preprocess function for input audio

import numpy as np
def preprocess
def debug_audio_data(audio_data):
    if isinstance(audio_data, np.ndarray):
        print(f'Audio data shape: {audio_data.shape}')
        print(f'Audio data type: {audio_data.dtype}')
        print(f'Audio data max: {np.max(audio_data)}')
        print(f'Audio data min: {np.min(audio_data)}')
    else:
        print('Audio data is not a numpy array')

    if audio_data is None or audio_data.size == 0:
        print('Audio data is empty or malformed')
    return audio_data

def preprocess(audio_data, max_duration=None):
    audio_data = debug_audio_data(audio_data)
(audio, max_duration):
    if isinstance(audio, tuple):
        y, sr = audio
        y = np.array(y, dtype=np.float32)  # Ensure audio data is floating-point
    else:
        y, sr = librosa.load(audio, sr=None)
        y = y.astype(np.float32)  # Ensure audio data is floating-point

    if y.ndim == 0 or y.size == 0:
        raise ValueError("Audio data must be at least one-dimensional and not empty")

    # Truncate or pad the audio to the max_duration
    max_samples = int(max_duration * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)), 'constant')

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    log_spectrogram = log_spectrogram[:, :128 * 128]  # Ensure the correct shape
    log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, 128 * 128 - log_spectrogram.shape[1])), 'constant')
    log_spectrogram = log_spectrogram[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions
    return torch.tensor(log_spectrogram, dtype=torch.float32)

# Postprocess function for output stems
def postprocess(stems):
    stems = stems.detach().cpu().numpy()
    return [stems[0, i, :] for i in range(stems.shape[1])]

# Audio separation function
def separate_audio(input_audio, model_checkpoint, checkpoint_path, max_duration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KANModel().to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, model_checkpoint), map_location=device))
    model.eval()
    input_data = preprocess(input_audio, max_duration).to(device)
    with torch.no_grad():
        separated_stems = model(input_data)
    output_stems = postprocess(separated_stems)
    return output_stems

# Refresh function to update model checkpoints
def refresh_checkpoints(checkpoint_path):
    return gr.Dropdown.update(choices=get_model_checkpoints(checkpoint_path))

# Gradio layout using Blocks
with gr.Blocks() as app:
    with gr.Tab("Train Model"):
        gr.Markdown("Train the Kolmogorov-Arnold Network model using stem data.")
        epochs = gr.Number(label='Epochs', value=10)
        learning_rate = gr.Number(label='Learning Rate', value=0.001)
        batch_size = gr.Number(label='Batch Size', value=8)  # Lowered default batch size to reduce memory usage
        dataset_path = gr.Textbox(label='Dataset Path', value='G:\\Music\\badmultitracks-michaeljackson\\dataset', placeholder='Enter dataset path')
        train_button = gr.Button("Train")
        train_output = gr.Textbox()
        train_button.click(train_model, inputs=[epochs, learning_rate, batch_size, dataset_path], outputs=train_output)

    with gr.Tab("Separate Audio"):
        gr.Markdown("Upload an audio file and get separated stems using Kolmogorov-Arnold Networks (KANs).")
        input_audio = gr.Audio(type='numpy')
        checkpoint_path = gr.Textbox(label='Checkpoint Path', value='C:\\projects\\KAN-Stem\\checkpoints', placeholder='Enter checkpoint path')
        model_checkpoint = gr.Dropdown(label='Model Checkpoint', choices=get_model_checkpoints('C:\\projects\\KAN-Stem\\checkpoints'), value='model.ckpt', interactive=True, allow_custom_value=True)
        max_duration = gr.Number(label='Max Audio Duration (seconds)', value=30)
        refresh_button = gr.Button("Refresh Checkpoints")
        refresh_button.click(fn=lambda: refresh_checkpoints(checkpoint_path.value), inputs=None, outputs=model_checkpoint)
        separate_button = gr.Button("Separate")
        output_stems = [gr.Audio(type='numpy') for _ in range(4)]
        separate_button.click(separate_audio, inputs=[input_audio, model_checkpoint, checkpoint_path, max_duration], outputs=output_stems)

if __name__ == '__main__':
    app.launch(server_name="127.0.0.1")
