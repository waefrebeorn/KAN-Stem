# KAN-Stem

This project uses Kolmogorov–Arnold Networks (KANs) for audio stem separation.

## Features

- Train a KAN model on audio stem data
- Separate audio into different stems using the trained KAN model

- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive)
- [cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive) (requires NVIDIA Developer account)

### Installation

1. Download and install CUDA 12.1.
2. Download cuDNN v8.9.7 and copy its contents to the CUDA installation directory:

```
Copy-Item -Path "C:\path\to\cudnn\bin\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
Copy-Item -Path "C:\path\to\cudnn\include\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
Copy-Item -Path "C:\path\to\cudnn\lib\x64\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64"
```

To set up the environment and install the dependencies, run:

python -m venv venv
source venv/bin/activate  # On Windows use \
env\Scripts\activate\
pip install -r requirements.txt


## Usage

To train the model, run:

python app.py

## Credits

This project incorporates code and ideas from the comprehensive [awesome-kan](https://github.com/mintisan/awesome-kan) repository. Special thanks to the authors for their contributions to the field.

This project incorporates code and ideas from the [KindXiaoming-pykan](https://github.com/KindXiaoming/pykan) repository. Special thanks to them for the initial arxiv paper.

