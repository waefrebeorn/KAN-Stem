import torch

def preprocess(audio_data):
    """
    Preprocess the audio data.

    Args:
    -----
    audio_data : numpy array or torch tensor
        The input audio data.

    Returns:
    --------
    preprocessed_data : torch tensor
        The preprocessed audio data.
    """
    print(f'Debug: Received audio data of type {type(audio_data)} and length {len(audio_data)}')
    if isinstance(audio_data, torch.Tensor):
        return audio_data
    else:
        return torch.tensor(audio_data)
