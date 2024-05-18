import gradio as gr
from main_script import train_model, separate_audio

train_interface = gr.Interface(
    fn=train_model,
    inputs=[gr.Number(label="Epochs"), gr.Number(label="Learning Rate")],
    outputs="text",
    title="Train KAN Model",
    description="Train the Kolmogorov-Arnold Network model using stem data."
)

separate_interface = gr.Interface(
    fn=separate_audio,
    inputs=gr.Audio(type="numpy"),
    outputs=[gr.Audio(type="numpy") for _ in range(4)],
    title="KAN Audio Stem Separation",
    description="Upload an audio file and get separated stems using Kolmogorov-Arnold Networks (KANs)."
)

app = gr.TabbedInterface(
    [train_interface, separate_interface],
    ["Train Model", "Separate Audio"]
)

if __name__ == "__main__":
    if not hasattr(gr, 'is_running'):
        gr.is_running = True
        app.launch()
