# KAN-Stem

KAN-Stem is an advanced project leveraging Kolmogorov–Arnold Networks (KANs) for state-of-the-art audio stem separation. This project aims to push the boundaries of audio processing and separation by integrating KANs with cutting-edge techniques and tools.

## Features

- **State-of-the-Art Model**: Train a KAN model specifically designed for audio stem separation, delivering superior performance.
- **Advanced Separation Techniques**: Separate audio into different stems using the trained KAN model with high precision and quality.
- **Optimized Performance**: Utilizes CUDA 12.1 and cuDNN v8.9.7 for accelerated training and inference.

## Installation

1. **Download and Install CUDA 12.1**
   - Visit [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive) and follow the instructions.

2. **Download cuDNN v8.9.7**
   - Visit [cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive) (requires NVIDIA Developer account).
   - Copy its contents to the CUDA installation directory:

```
Copy-Item -Path "C:\path\to\cudnn\bin\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
Copy-Item -Path "C:\path\to\cudnn\include\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
Copy-Item -Path "C:\path\to\cudnn\lib\x64\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64"
```

3. **Set up the environment and install dependencies**

```
python -m venv venv
source venv/bin/activate  # On Windows use env\Scripts\activate
pip install -r requirements.txt
```

## Usage

To train the model, run:

```
python gradio_app.py
```

## Credits

This project integrates the combined efforts and innovations from various researchers and repositories in the field of deep learning and audio processing. Special thanks to the following contributors:

- [awesome-kan](https://github.com/mintisan/awesome-kan) repository for their comprehensive resources on Kolmogorov–Arnold Networks.
- [KindXiaoming-pykan](https://github.com/KindXiaoming/pykan) for the initial implementation and the insightful arXiv paper.
- [facebookresearch/demucs](https://github.com/facebookresearch/demucs) for their groundbreaking work in music source separation which influenced this project.
- The authors of the [Kolmogorov–Arnold Network paper](https://arxiv.org/pdf/2404.19756) for their foundational research that inspired this project.
- [Optuna](https://github.com/optuna/optuna) and [Ray Tune](https://github.com/ray-project/ray) for their exceptional hyperparameter optimization tools used in this project.
- The [Gradio](https://gradio.app/) team for providing a user-friendly interface for training and testing the model.
- [Librosa](https://github.com/librosa/librosa) for their Python package for music and audio analysis.
- [PyTorch](https://pytorch.org/) for their deep learning framework that serves as the backbone of this project.
- [Scikit-learn](https://scikit-learn.org/) for their machine learning tools which were utilized for various preprocessing tasks.
- [NumPy](https://numpy.org/) for their fundamental package for scientific computing with Python.
- [Pandas](https://pandas.pydata.org/) for their data analysis and manipulation tools.
- [SciPy](https://www.scipy.org/) for their ecosystem of open-source software for mathematics, science, and engineering.
- [Matplotlib](https://matplotlib.org/) for their plotting library used for visualizing results.
- [Seaborn](https://seaborn.pydata.org/) for their data visualization library based on Matplotlib.
- [TensorBoard](https://www.tensorflow.org/tensorboard) for providing the tools to visualize and debug the training process.
- **Orion** for their valuable tools and resources that supported this project.
- **Gemini** for web search and instructional guidance.
- **Claude Opus** for logical assistance during challenging phases of the project.
- **ChatGPT** for hybridizing and coding, ensuring the successful implementation and integration of various components.

## Additional Information

- **Performance Metrics**: The project evaluates model performance using SDR, SIR, and SAR metrics to ensure high-quality audio separation.
- **Hyperparameter Optimization**: Utilizes Optuna and Ray Tune for efficient and effective hyperparameter optimization.
- **Logging and Visualization**: Integrated with TensorBoard for detailed logging and visualization of training progress and results.

By incorporating these state-of-the-art techniques and tools, KAN-Stem sets a new benchmark in the field of audio stem separation. The project aims to provide researchers and developers with a powerful tool to achieve high-quality audio separation with minimal effort.

## License

This project is licensed under the MIT License.

---

For any further questions or contributions, feel free to open an issue or pull request on our GitHub repository.

---
