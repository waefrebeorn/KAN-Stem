# KAN-Stem

This project uses Kolmogorov–Arnold Networks (KANs) for audio stem separation.

## Features

- Train a KAN model on audio stem data
- Separate audio into different stems using the trained KAN model
## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/KAN-Stem.git
    cd KAN-Stem
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3. Run the setup script:
    ```sh
    .\setup.ps1
    ```

This will install the necessary dependencies, run the training script, and launch TensorBoard for performance monitoring.

## Requirements

- Python 3.10
- CUDA 12.1

The following dependencies will be installed:
- torch
- torchvision
- torchaudio
- numpy
- pandas
- tensorflow
- gradio
- python-dateutil
- pytz
- tzdata
- tensorflow-intel
- keras
- typing-extensions

## Installation

To set up the environment and install the dependencies, run:

\\\ash
python -m venv venv
source venv/bin/activate  # On Windows use \
env\Scripts\activate\
pip install -r requirements.txt
\\\

## Usage

To train the model, run:

\\\ash
python app.py
\\\

## Credits

This project incorporates code and ideas from the [efficient-kan](https://github.com/Blealtan/efficient-kan) repository. Special thanks to the authors for their contributions to the field.

## License

[MIT License](LICENSE)

## Credits
This project incorporates the KAN model from [KindXiaoming's pykan](https://github.com/KindXiaoming/pykan).

This project incorporates the KAN model from [Blealtan's efficient-kan](https://github.com/Blealtan/efficient-kan).

