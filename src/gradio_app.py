import os
import threading
import time
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import gradio as gr
import subprocess
import soundfile as sf
from scipy.interpolate import BSpline
from pathlib import Path
import psutil
import gc

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

print("Starting Gradio interface with KAN functionality...")

# Function to print current memory usage
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{step}: Memory usage: {mem_info.rss / 1024**2:.2f} MB")

class SplineActivation(nn.Module):
    def __init__(self, num_params=1, num_knots=10):
        super(SplineActivation, self).__init__()
        self.num_params = num_params
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots))
        self.coeffs = nn.Parameter(torch.ones(num_knots, num_params))

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, self.num_knots)
        x = torch.bmm(x, self.coeffs.unsqueeze(0).expand(x.size(0), -1, -1))
        return x.sum(dim=-1)

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super(KANLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = SplineActivation(num_params=output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class KANModel(nn.Module):
    def __init__(self, input_size, num_layers=3, degree=3):
        super(KANModel, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for _ in range(num_layers - 1):
            self.layers.append(KANLayer(current_size, current_size * 2, degree))
            current_size = current_size * 2
        self.layers.append(KANLayer(current_size, 7, degree))  # Output 7 stems

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def ensure_mono(waveform):
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform

def normalize_length(waveform, sample_rate, target_length):
    target_samples = int(target_length * sample_rate)
    current_samples = waveform.shape[0]
    
    if current_samples > target_samples:
        return waveform[:target_samples]
    else:
        repeats = target_samples // current_samples
        remainder = target_samples % current_samples
        return np.concatenate([np.tile(waveform, repeats), waveform[:remainder]])

def preprocess_audio(audio_file_path, target_length=180):
    print(f"Preprocessing audio file: {audio_file_path}")
    try:
        waveform, sample_rate = sf.read(audio_file_path, dtype='float32')
        waveform = ensure_mono(waveform)
        waveform = normalize_length(waveform, sample_rate, target_length)
        waveform = torch.tensor(waveform)

        chunks = torch.split(waveform, 44100)

        return chunks, sample_rate
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None

def predict(model, input_data):
    print(f"Running prediction with model: {model} on input data: {input_data.shape}")
    try:
        input_data = input_data.T
        input_data = input_data.unsqueeze(0)
        input_data = input_data.cuda() if torch.cuda.is_available() else input_data
        model = model.cuda() if torch.cuda.is_available() else model.cpu()
        output = model(input_data)
        print(f"Prediction output: {output.shape}")
        return output.detach().cpu().numpy()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def run_prediction(audio_chunks):
    model = KANModel(input_size=44100, num_layers=3)
    checkpoint_dir = "checkpoints"
    checkpoints = [f for f in Path(checkpoint_dir).iterdir() if f.is_file() and f.name.startswith("kan_model_checkpoint_")]
    if checkpoints:
        best_checkpoint = sorted(checkpoints, key=lambda p: sum(float(w) for w in p.name.split("_")[2:-1]))[-1]
        model.load_state_dict(torch.load(best_checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"Loaded checkpoint with winners: {best_checkpoint.name}")
    else:
        print("No checkpoint found. Please train the model first.")
        return None
    
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()
    outputs = []
    for chunk in audio_chunks:
        chunk = chunk.cuda() if torch.cuda.is_available() else chunk
        output = predict(model, chunk)
        if output is not None:
            outputs.append(output)
    if len(outputs) == 0:
        return None
    return np.concatenate(outputs, axis=2)

def split_audio(audio_file_path):
    try:
        audio_chunks, sample_rate = preprocess_audio(audio_file_path)
        if audio_chunks is None:
            raise Exception("Error during audio preprocessing")
        output = run_prediction(audio_chunks)
        if output is None:
            raise Exception("Error during prediction")
        stems = []
        for i in range(output.shape[1]):
            stem_path = f"stem_{i+1}.wav"
            sf.write(stem_path, output[0, i], sample_rate)
            stems.append(stem_path)
        return stems
    except Exception as e:
        print(f"Error during audio splitting: {e}")
        return [None] * 5

training_process = None
training_start_time = None

def monitor_training():
    global training_process
    try:
        while True:
            output = training_process.stdout.readline()
            if output == b'' and training_process.poll() is not None:
                break
            if output:
                print(output.strip().decode())
                print_memory_usage("During training")
    except Exception as e:
        print(f"Error monitoring training: {e}")

def start_training(dataset_path, num_epochs, batch_size, learning_rate, chunk_size, hop_length, n_fft):
    global training_process, training_start_time
    try:
        training_start_time = time.time()
        use_cuda_flag = "--use_cuda" if torch.cuda.is_available() else ""
        print(f"Starting training with dataset={dataset_path}, num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, chunk_size={chunk_size}, hop_length={hop_length}, n_fft={n_fft}")
        training_process = subprocess.Popen(
            ["python", "src/train.py", "--dataset", dataset_path, "--num_epochs", str(num_epochs), "--batch_size", str(batch_size), "--learning_rate", str(learning_rate), "--chunk_size", str(chunk_size), "--hop_length", str(hop_length), "--n_fft", str(n_fft), use_cuda_flag],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        monitor_thread = threading.Thread(target=monitor_training)
        monitor_thread.start()
        return "Training started successfully. Check the terminal for detailed updates."
    except subprocess.CalledProcessError as e:
        return f"Error starting training: {e}"

def training_status():
    if training_process and training_process.poll() is None:
        elapsed_time = time.time() - training_start_time
        return f"Training in progress... Elapsed time: {int(elapsed_time)} seconds"
    else:
        return "No training in progress."

def prepare_dataset(input_dir, output_dir, num_examples):
    try:
        print(f"Preparing dataset with input_dir={input_dir}, output_dir={output_dir}, num_examples={num_examples}")
        subprocess.check_call([
            "python", "src/prepare_dataset.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--num_examples", str(num_examples)
        ])
        return f"Dataset prepared with {num_examples} examples in {output_dir}."
    except subprocess.CalledProcessError as e:
        return f"Error preparing dataset: {e}"

def process_to_dataset(input_dir, output_dir, num_examples):
    try:
        print(f"Processing to dataset with input_dir={input_dir}, output_dir={output_dir}, num_examples={num_examples}")
        subprocess.check_call([
            "python", "src/process_to_dataset.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--num_examples", str(num_examples)
        ])
        return f"Dataset processed with {num_examples} examples in {output_dir}."
    except subprocess.CalledProcessError as e:
        return f"Error processing dataset: {e}"

def calculate_num_examples(memory_gb, duration_sec=180, num_stems=5, sample_rate=44100):
    num_samples = duration_sec * sample_rate
    example_size_bytes = num_samples * num_stems * 4
    
    memory_bytes = memory_gb * 1024**3
    
    num_examples = memory_bytes // (example_size_bytes * 1.5)
    
    return int(num_examples)

default_num_examples = calculate_num_examples(32)

def create_gradio_interface():
    with gr.Blocks() as interface:
        print("Setting up Gradio interface with KAN functionality...")

        with gr.Tab("Audio Splitter"):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (FLAC or WAV)")
            output_stems = [gr.Audio(label=f"Stem {i+1}") for i in range(5)]
            gr.Button("Run").click(
                fn=split_audio,
                inputs=audio_input,
                outputs=output_stems
            )

        with gr.Tab("Training"):
            dataset_path = gr.Textbox(label="Dataset Path", value="K:\\KAN-Stem DataSet\\Chunks\\Chunk_0_Sample")
            num_epochs = gr.Number(label="Number of Epochs", value=50)
            batch_size = gr.Number(label="Batch Size", value=1)
            learning_rate = gr.Number(label="Learning Rate", value=0.001)
            chunk_size = gr.Number(label="Chunk Size", value=256)
            hop_length = gr.Number(label="Hop Length", value=128)
            n_fft = gr.Number(label="Number of FFT Points", value=512)
            start_button = gr.Button("Start Training")
            status_box = gr.Markdown("No training in progress.")
            
            start_button.click(
                fn=start_training,
                inputs=[dataset_path, num_epochs, batch_size, learning_rate, chunk_size, hop_length, n_fft],
                outputs=None
            )
            
            interface.load(fn=training_status, inputs=None, outputs=status_box, every=1)

        with gr.Tab("Dataset Preparation"):
            input_dir = gr.Textbox(label="Input Directory", value="K:\\KAN-Stem DataSet")
            output_dir = gr.Textbox(label="Output Directory", value="K:\\KAN-Stem DataSet\\Kan-StemRB1")
            num_examples = gr.Number(label="Number of Examples", value=default_num_examples)
            prepare_button = gr.Button("Prepare Dataset")
            output_message = gr.Textbox(label="Output Message")
            prepare_button.click(prepare_dataset, inputs=[input_dir, output_dir, num_examples], outputs=output_message)

        with gr.Tab("Process to Dataset"):
            input_dir = gr.Textbox(label="Input Directory", value="K:\\KAN-Stem DataSet\\Kan-StemRB1")
            output_dir = gr.Textbox(label="Output Directory", value="K:\\KAN-Stem DataSet\\ProcessedDataset")
            num_examples = gr.Number(label="Number of Examples", value=default_num_examples)
            process_button = gr.Button("Process to Dataset")
            output_message = gr.Textbox(label="Output Message")
            process_button.click(process_to_dataset, inputs=[input_dir, output_dir, num_examples], outputs=output_message)

    return interface

interface = create_gradio_interface()

if __name__ == "__main__":
    try:
        print("Launching Gradio interface with KAN functionality...")
        interface.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
        print("Gradio interface should now be running on http://127.0.0.1:7860")
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
