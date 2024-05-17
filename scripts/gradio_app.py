import gradio as gr
from audio_stem_separation import process_audio

def gradio_interface(file):
    processed_data = process_audio(file.name)
    return processed_data

iface = gr.Interface(fn=gradio_interface, inputs="file", outputs="plot")
iface.launch()
