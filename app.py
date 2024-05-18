import os
import gradio as gr
import torch
import librosa
import numpy as np
from modules import KANModel, preprocess, postprocess
import torch.optim as optim
import torch.nn as nn

# Function to load data from a given directory
def load_stem_data(dataset_path):
    wav_path = os.path.join(dataset_path, 'wav')
    wav_files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]

    inputs = []
    targets = []

    for wav_file in wav_files:
        y, sr = librosa.load(wav_file, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        inputs.append(log_spectrogram)
        
        # Example targets, in practice, load actual target data
        target = np.random.randn(4, log_spectrogram.shape[1])  # Example target with 4 stems
        targets.append(target)
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

# Training function
def train_model(epochs, learning_rate, batch_size, dataset_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs, targets = load_stem_data(dataset_path)
    model = KANModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (len(inputs) // batch_size)}')

    torch.save(model.state_dict(), 'checkpoints/model.ckpt')
    return 'Model trained and saved at checkpoints/model.ckpt'

# Audio separation function
def separate_audio(input_audio):
    model = KANModel().load_from_checkpoint('checkpoints/model.ckpt')
    input_data = preprocess(input_audio)
    separated_stems = model(input_data)
    output_stems = postprocess(separated_stems)
    return output_stems

# Gradio interfaces
train_interface = gr.Interface(
    fn=train_model,
    inputs=[
        gr.Number(label='Epochs', value=10),
        gr.Number(label='Learning Rate', value=0.001),
        gr.Number(label='Batch Size', value=32),
        gr.Textbox(label='Dataset Path', value='G:\\Music\\badmultitracks-michaeljackson\\dataset', placeholder='Enter dataset path')
    ],
    outputs='text',
    title='Train KAN Model',
    description='Train the Kolmogorov-Arnold Network model using stem data.'
)

separate_interface = gr.Interface(
    fn=separate_audio,
    inputs=gr.Audio(type='numpy'),
    outputs=[gr.Audio(type='numpy') for _ in range(4)],
    title='KAN Audio Stem Separation',
    description='Upload an audio file and get separated stems using Kolmogorov-Arnold Networks (KANs).'
)

app = gr.TabbedInterface(
    [train_interface, separate_interface],
    ['Train Model', 'Separate Audio']
)

if __name__ == '__main__':
    if not hasattr(gr, 'is_running'):
        gr.is_running = True
        app.launch()
