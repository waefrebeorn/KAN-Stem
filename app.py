import os
    app.launch(server_name="127.0.0.1")
import os
    app.launch(server_name="127.0.0.1")
import os
    app.launch(server_name="127.0.0.1")
if __name__ == '__main__':
    app.launch(server_name="127.0.0.1")
import os
import gradio as gr
import torch
import librosa
import numpy as np
# import requests  # Commenting out for now
from modules.KANModel import KANModel  # Ensure correct import
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Function to load data from a given directory
def load_stem_data(dataset_path):
    try:
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
    except Exception as e:
        print(f"Error in load_stem_data: {e}")
        raise

# Training function
def train_model(epochs, learning_rate, batch_size, dataset_path):
    try:
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
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise

# Function to get a list of model checkpoints
def get_model_checkpoints(checkpoint_path):
    try:
        return [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
    except Exception as e:
        print(f"Error in get_model_checkpoints: {e}")
        raise

# Function to download model checkpoint from Hugging Face
# def download_checkpoint(url, checkpoint_path):
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
#     response = requests.get(url)
#     with open(os.path.join(checkpoint_path, 'model.ckpt'), 'wb') as f:
#         f.write(response.content)

# Preprocess function for input audio
def preprocess(audio, max_duration):
# Preprocess function for input audio

#         f.write(response.content)
#     with open(os.path.join(checkpoint_path, 'model.ckpt'), 'wb') as f:
#     response = requests.get(url)
#         os.makedirs(checkpoint_path)
#     if not os.path.exists(checkpoint_path):
# def download_checkpoint(url, checkpoint_path):
# Function to download model checkpoint from Hugging Face

        raise
        print(f"Error in get_model_checkpoints: {e}")
    except Exception as e:
        return [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
    try:
def get_model_checkpoints(checkpoint_path):
# Function to get a list of model checkpoints

        raise
        print(f"Error in train_model: {e}")
    except Exception as e:
        return 'Model trained and saved at checkpoints/model.ckpt'
        torch.save(model.state_dict(), os.path.join('checkpoints', 'model.ckpt'))
        writer.close()

            writer.add_scalar('Loss/train', avg_loss, epoch)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
            avg_loss = total_loss / (len(inputs) // batch_size)
                total_loss += loss.item()
                optimizer.step()
                loss.backward()
                loss = criterion(output, target_batch)
                output = model(input_batch)
                optimizer.zero_grad()
                target_batch = torch.tensor(target_batch, dtype=torch.float32).to(device)
                input_batch = torch.tensor(input_batch, dtype=torch.float32).to(device)
                target_batch = targets[i:i + batch_size]
                input_batch = inputs[i:i + batch_size]
            for i in range(0, len(inputs), batch_size):
            total_loss = 0
            model.train()
        for epoch in range(epochs):

        writer = SummaryWriter()
        # TensorBoard writer

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model = KANModel().to(device)
        inputs, targets = load_stem_data(dataset_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
def train_model(epochs, learning_rate, batch_size, dataset_path):
# Training function

        raise
        print(f"Error in load_stem_data: {e}")
    except Exception as e:
        return inputs, targets

        inputs = inputs[:, np.newaxis, :, :]  # Add a channel dimension
        # Ensure inputs have the correct shape

        targets = np.array(targets)
        inputs = np.array(inputs)

            targets.append(target)
            target = np.random.randn(4, 44100)  # Example target with 4 stems
            # Example targets, in practice, load actual target data

            inputs.append(log_spectrogram)

                log_spectrogram = log_spectrogram[:, :target_shape[1]]
            else:
                log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, target_shape[1] - log_spectrogram.shape[1])), 'constant')
            if log_spectrogram.shape[1] < target_shape[1]:
            target_shape = (128, 128 * 128)
            # Ensure the log_spectrogram has the correct shape

            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            y = y.astype(np.float32)  # Ensure audio data is floating-point
            y, sr = librosa.load(wav_file, sr=None)
        for wav_file in wav_files:

        targets = []
        inputs = []

        wav_files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]
        wav_path = os.path.join(dataset_path, 'wav')
    try:
def load_stem_data(dataset_path):
# Function to load data from a given directory

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from modules.KANModel import KANModel  # Ensure correct import
# import requests  # Commenting out for now
import numpy as np
import librosa
import torch
import gradio as gr
import os
    app.launch(server_name="127.0.0.1")
if __name__ == '__main__':
    app.launch(server_name="127.0.0.1")
import os
    app.launch(server_name="127.0.0.1")
import os
    app.launch(server_name="127.0.0.1")
import os
    app.launch(server_name="127.0.0.1")
if __name__ == '__main__':
    app.launch(server_name="127.0.0.1")
import os
import gradio as gr
import torch
import librosa
import numpy as np
# import requests  # Commenting out for now
from modules.KANModel import KANModel  # Ensure correct import
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Function to load data from a given directory
def load_stem_data(dataset_path):
    try:
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
    except Exception as e:
        print(f"Error in load_stem_data: {e}")
        raise

# Training function
def train_model(epochs, learning_rate, batch_size, dataset_path):
    try:
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
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise

# Function to get a list of model checkpoints
def get_model_checkpoints(checkpoint_path):
    try:
        return [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
    except Exception as e:
        print(f"Error in get_model_checkpoints: {e}")
        raise

# Function to download model checkpoint from Hugging Face
# def download_checkpoint(url, checkpoint_path):
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
#     response = requests.get(url)
#     with open(os.path.join(checkpoint_path, 'model.ckpt'), 'wb') as f:
#         f.write(response.content)

# Preprocess function for input audio
def preprocess(audio, max_duration):
    try:
        print(f"Input audio: {audio}")  # Debug statement
        if isinstance(audio, tuple):
            y, sr = audio
            y = np.array(y, dtype=np.float32)  # Ensure audio data is floating-point
        else:
            file_ext = os.path.splitext(audio)[-1].lower()
            if file_ext == '.flac':
                y, sr = librosa.load(audio, sr=None, mono=True)
            else:
                y, sr = librosa.load(audio, sr=None)
            y = y.astype(np.float32)  # Ensure audio data is floating-point

        print(f"Audio shape: {y.shape}, Sample rate: {sr}")  # Debug statement

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
    except Exception as e:
        print(f"Error in preprocess: {e}")
        raise

# Postprocess function for output stems
def postprocess(stems):
    try:
        stems = stems.detach().cpu().numpy()
        return [stems[0, i, :] for i in range(stems.shape[1])]
    except Exception as e:
        print(f"Error in postprocess: {e}")
        raise

# Audio separation function
def separate_audio(input_audio, model_checkpoint, checkpoint_path, max_duration):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = KANModel().to(device)
        model_checkpoint_path = os.path.join(checkpoint_path, model_checkpoint)
        # if not os.path.exists(model_checkpoint_path):
        #     download_checkpoint('https://huggingface.co/WaefreBeorn/KAN-Stem/blob/main/model.ckpt', checkpoint_path)
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        model.eval()
        input_data = preprocess(input_audio, max_duration).to(device)
        with torch.no_grad():
            separated_stems = model(input_data)
        output_stems = postprocess(separated_stems)
        return output_stems
    except Exception as e:
        print(f"Error in separate_audio: {e}")
        raise

# Refresh function to update model checkpoints
def refresh_checkpoints(checkpoint_path):
    try:
        return gr.Dropdown.update(choices=get_model_checkpoints(checkpoint_path))
    except Exception as e:
        print(f"Error in refresh_checkpoints: {e}")
        raise

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
