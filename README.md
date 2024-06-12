# KAN-Stem

**KAN-Stem** is a cutting-edge audio stem separation project leveraging the power of **Kolmogorov–Arnold Networks (KANs)** and incorporating the latest advancements from the paper ["On the Representational Power of the Kolmogorov-Arnold Network"](https://arxiv.org/html/2312.10949v1). The goal is to set new standards in audio processing and separation by integrating KANs with state-of-the-art techniques and tools.

## Key Features

- **State-of-the-Art Model:**  The project trains a KAN model specifically tailored for audio stem separation, enabling superior performance and accuracy.
- **Advanced Separation Techniques:** Utilizes the trained KAN model to separate audio into distinct stems (vocals, drums, bass, etc.) with high precision and quality.
- **Optimized Performance:** Leverages CUDA 12.1 and cuDNN v8.9.7 for accelerated training and inference, ensuring efficient and fast processing.
- **Novel KAN Architecture:** Incorporates architectural improvements inspired by ["On the Representational Power of the Kolmogorov-Arnold Network"](https://arxiv.org/html/2312.10949v1) to enhance the model's capacity for audio representation and separation. 

## Installation

1. **CUDA 12.1 and cuDNN v8.9.7:**
   - Download and install CUDA 12.1 from [https://developer.nvidia.com/cuda-12-1-1-download-archive](https://developer.nvidia.com/cuda-12-1-1-download-archive).
   - Download cuDNN v8.9.7 from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive) (requires NVIDIA Developer account).
   - Copy cuDNN contents to the CUDA installation directory.

2. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

- **Training:**
  ```bash
  python gradio_app.py
  ```
  This launches a user-friendly Gradio interface for training the model, visualizing results, and experimenting with different settings.

- **Inference:**
  ```bash
  python infer.py --input_audio_path <path_to_your_audio_file> --output_path <path_to_save_separated_stems> 
  ```
  This command takes an audio file as input and generates the separated stems in the specified output directory.

## Expanded Credits

KAN-Stem builds upon the collective knowledge and innovations from a wide range of contributors:

- **Research and Papers:**
  - The authors of ["Leveraged Mel spectrograms using Harmonic and Percussive Components in Speech Emotion Recognition"](https://arxiv.org/html/2312.10949v1) for their significant contributions to KAN theory and architecture.
  - The original Kolmogorov–Arnold Network paper ([https://arxiv.org/pdf/2404.19756](https://arxiv.org/pdf/2404.19756)) for laying the foundation for this project.
  - [facebookresearch/demucs](https://github.com/facebookresearch/demucs) for their influential work in music source separation. 
  - [awesome-kan](https://github.com/mintisan/awesome-kan) for their comprehensive resources on KANs.

- **Implementations:**
  - [KindXiaoming-pykan](https://github.com/KindXiaoming/pykan) for their initial implementation and insightful arXiv paper.

- **Tools and Libraries:**
  - [Optuna](https://github.com/optuna/optuna) and [Ray Tune](https://github.com/ray-project/ray) for their powerful hyperparameter optimization capabilities.
  - [Gradio](https://gradio.app/) for simplifying the model training and testing experience with an intuitive interface.
  - [Librosa](https://github.com/librosa/librosa) for audio analysis and manipulation.
  - [PyTorch](https://pytorch.org/) for the deep learning framework.
  - [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) for machine learning, data analysis, and visualization.
  - [TensorBoard](https://www.tensorflow.org/tensorboard) for monitoring and debugging the training process.

- **AI Assistants:**
  - Orion, Gemini, Claude Opus, and ChatGPT for their invaluable assistance in research, coding, debugging, and optimization throughout the project's development.

## Additional Notes

- **Performance Metrics:**  SDR, SIR, and SAR metrics are used to rigorously assess the model's audio separation quality.
- **Hyperparameter Optimization:** Optuna and Ray Tune are employed for efficient and thorough hyperparameter tuning, leading to optimal model performance.
- **Logging and Visualization:** TensorBoard integration allows for comprehensive tracking and visualization of training progress and results.


## License

This project is licensed under the MIT License. Feel free to contribute and improve KAN-Stem by opening issues or submitting pull requests on our GitHub repository.
