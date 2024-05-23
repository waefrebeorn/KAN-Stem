import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

print("Starting Gradio interface with KAN functionality...")

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree):
        super(KANLayer, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(degree)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(degree)])
        self.degree = degree

    def forward(self, x):
        out = self.linears[0](x)
        for i in range(1, self.degree):
            out = out + self.activations[i](self.linears[i](x))
        return out

class KANModel(nn.Module):
    def __init__(self, degree=3):
        super(KANModel, self).__init__()
        self.kan1 = KANLayer(44100, 64, degree)
        self.kan2 = KANLayer(64, 128, degree)
        self.kan3 = KANLayer(128, 256, degree)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 4 * 44100)  # Adjust for 4 output stems

    def forward(self, x):
        x = F.relu(self.kan1(x))
        x = F.relu(self.kan2(x))
        x = F.relu(self.kan3(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), 4, 44100)  # Reshape to (batch_size, 4, 44100)

def preprocess_audio(audio_file_path, chunk_size=44100):
    print(f"Preprocessing audio file: {audio_file_path}")
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
        num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size
        waveform = F.pad(waveform, (0, padded_length - waveform.shape[1]))
        chunks = torch.split(waveform, chunk_size, dim=1)
        print(f"Preprocessed audio data: {waveform.shape}, Chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def predict(model, input_data):
    print(f"Running prediction with model: {model} on input data: {input_data.shape}")
    try:
        input_data = input_data.unsqueeze(0)  # Add batch dimension
        output = model(input_data)
        print(f"Prediction output: {output.shape}")
        return output.detach().cpu().numpy()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def run_prediction(audio_chunks):
    model = KANModel()
    checkpoint_path = os.path.join("checkpoints", "model_epoch_50.ckpt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    outputs = []
    for chunk in audio_chunks:
        output = predict(model, chunk)
        if output is not None:
            outputs.append(output)
    if len(outputs) == 0:
        return None
    return np.concatenate(outputs, axis=2)

def split_audio(audio_file_path):
    try:
        audio_chunks = preprocess_audio(audio_file_path)
        if audio_chunks is None:
            raise Exception("Error during audio preprocessing")
        output = run_prediction(audio_chunks)
        if output is None:
            raise Exception("Error during prediction")
        stems = []
        for i in range(output.shape[1]):
            stem_path = f"stem_{i+1}.wav"
            torchaudio.save(stem_path, torch.tensor(output[0, i]), 44100)
            stems.append(stem_path)
        return stems
    except Exception as e:
        print(f"Error during audio splitting: {e}")
        return [None] * 4

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks() as interface:
        print("Setting up Gradio interface with KAN functionality...")
        audio_input = gr.Audio(type="filepath", label="Upload Audio File (FLAC or WAV)")
        output_stems = [gr.Audio(label=f"Stem {i+1}") for i in range(4)]
        gr.Button("Run").click(
            fn=split_audio,
            inputs=audio_input,
            outputs=output_stems
        )
    return interface

# Create and launch the Gradio interface
interface = create_gradio_interface()

if __name__ == "__main__":
    try:
        print("Launching Gradio interface with KAN functionality...")
        interface.launch(server_name="127.0.0.1", show_error=True)
        print("Gradio interface should now be running on http://127.0.0.1:7860")
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
