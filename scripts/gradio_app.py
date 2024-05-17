import gradio as gr
from scripts.audio_stem_separation import load_audio, separate_stems, save_stems


def process_audio(file):
    audio, sr = load_audio(file.name)
    stems = separate_stems(audio)
    output_dir = "output"
    save_stems(stems, sr, output_dir)
    return [f"{output_dir}/vocals.wav", f"{output_dir}/accompaniment.wav"]


interface = gr.Interface(
    fn=process_audio,
    inputs=gr.inputs.Audio(source="upload", type="file"),
    outputs=[gr.outputs.Audio(type="file"), gr.outputs.Audio(type="file")],
    title="KAN-Stem Audio Separation",
    description="Upload an audio file to separate its stems using KANs.",
)

if __name__ == "__main__":
    interface.launch()
import gradio as gr
from audio_stem_separation import process_audio

def gradio_interface(file):
    processed_data = process_audio(file.name)
    return processed_data

iface = gr.Interface(fn=gradio_interface, inputs="file", outputs="plot")
iface.launch()
