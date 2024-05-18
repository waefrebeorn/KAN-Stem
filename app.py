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
    torch.save(model.state_dict(), 'checkpoints/model.ckpt')
    return 'Model trained and saved at checkpoints/model.ckpt'

# Function to get a list of model checkpoints
def get_model_checkpoints():
    checkpoint_path = 'checkpoints'
    return [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]

# Preprocess function for input audio
def preprocess(audio):
    if isinstance(audio, tuple):
        y, sr = audio
        y = np.array(y, dtype=np.float32)  # Ensure audio data is floating-point
    else:
        y, sr = librosa.load(audio, sr=None)
        y = y.astype(np.float32)  # Ensure audio data is floating-point

    if y.ndim == 0 or y.size == 0:
        raise ValueError("Audio data must be at least one-dimensional and not empty")

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
def separate_audio(input_audio, model_checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KANModel().to(device)
    model.load_state_dict(torch.load(f'checkpoints/{model_checkpoint}', map_location=device))
    model.eval()
    input_data = preprocess(input_audio).to(device)
    with torch.no_grad():
        separated_stems = model(input_data)
    output_stems = postprocess(separated_stems)
    return output_stems

# Gradio interfaces
train_interface = gr.Interface(
    fn=train_model,
    inputs=[
        gr.Number(label='Epochs', value=10),
        gr.Number(label='Learning Rate', value=0.001),
        gr.Number(label='Batch Size', value=8),  # Lowered default batch size to reduce memory usage
        gr.Textbox(label='Dataset Path', value='G:\\Music\\badmultitracks-michaeljackson\\dataset', placeholder='Enter dataset path')
    ],
    outputs='text',
    title='Train KAN Model',
    description='Train the Kolmogorov-Arnold Network model using stem data.'
)

separate_interface = gr.Interface(
    fn=separate_audio,
    inputs=[
        gr.Audio(type='numpy'),
        gr.Dropdown(label='Model Checkpoint', choices=get_model_checkpoints(), value='model.ckpt', interactive=True)
    ],
    outputs=[gr.Audio(type='numpy') for _ in range(4)],
    title='KAN Audio Stem Separation',
    description='Upload an audio file and get separated stems using Kolmogorov-Arnold Networks (KANs).',
    refresh_button=True
)

app = gr.TabbedInterface(
    [train_interface, separate_interface],
    ['Train Model', 'Separate Audio']
)

if __name__ == '__main__':
    if not hasattr(gr, 'is_running'):
        gr.is_running = True
        app.launch()
