import torch
import torchaudio
import logging
from torchaudio import transforms as T

logger = logging.getLogger(__name__)

def load_and_preprocess(file_path, mel_spectrogram, target_length, apply_data_augmentation, device):
    try:
        logger.debug("Starting load and preprocess")
        input_audio, _ = torchaudio.load(file_path)
        if input_audio is None:
            return None

        logger.debug(f"Device for processing: {device}")

        if apply_data_augmentation:
            input_audio = input_audio.float().to(device)
            logger.debug(f"input_audio device: {input_audio.device}")
            input_audio = data_augmentation(input_audio, device=device)
            logger.debug(f"After data augmentation, input_audio device: {input_audio.device}")
        else:
            input_audio = input_audio.float().to(device)

        input_mel = mel_spectrogram(input_audio).squeeze(0)[:, :target_length]

        # Move input_mel back to CPU after processing on GPU
        input_mel = input_mel.to('cpu')
        logger.debug(f"input_mel device: {input_mel.device}")

        logger.debug("Completed load and preprocess")
        return input_mel

    except Exception as e:
        logger.error(f"Error in load and preprocess: {e}")
        return None

def data_augmentation(inputs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2).to(device)
        freq_mask = T.FrequencyMasking(freq_mask_param=15).to(device)
        time_mask = T.TimeMasking(time_mask_param=35).to(device)

        augmented_inputs = pitch_shift(inputs.clone().detach().to(device))
        augmented_inputs = freq_mask(augmented_inputs.clone().detach())
        augmented_inputs = time_mask(augmented_inputs.clone().detach())

        return augmented_inputs
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        return inputs
