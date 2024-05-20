import os
import gradio as gr
import torch
from modules import KANModel, preprocess, postprocess

def separate_audio(input_audio):
    model = KANModel.load_from_checkpoint("checkpoints/model.ckpt")
    input_data = preprocess(input_audio)
    separated_stems = model(input_data)
    output_stems = postprocess(separated_stems)
    return output_stems

iface = gr.Interface(
    fn=separate_audio,
    inputs=gr.Audio\(sources=\["upload"\], type="numpy"),
    outputs=[gr.outputs.Audio(type="numpy") for _ in range(4)],
    title="KAN Audio Stem Separation",
    description="Upload an audio file and get separated stems using Kolmogorov-Arnold Networks (KANs)."
)

if __name__ == "__main__":
    iface.launch()
