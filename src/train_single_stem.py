import os
import logging
import tempfile
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import KANWithDepthwiseConv, KANDiscriminator
from cached_dataset import StemSeparationDataset
from utils import detect_parameters
from log_setup import logger
import gc

def train_single_stem(stem, data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_workers, cache_dir):
    logger.info(f"Training model for stem: {stem}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256  # Define your target length here

    dataset = StemSeparationDataset(data_dir, n_mels=n_mels, target_length=target_length, n_fft=n_fft, num_stems=7, cache_dir=cache_dir)
    logger.info(f"Number of valid file sets (input + corresponding stems): {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("No valid audio files found in the dataset after filtering.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False, num_workers=num_workers)

    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=7, cache_dir=cache_dir).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length).to(device)
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets_list = data
            logger.debug(f"Type of inputs: {type(inputs)}, Type of targets_list: {type(targets_list)}")
            print(f'Batch {i} inputs shape: {inputs.shape}')  # Debug print
            print(f'Batch {i} targets shape: {[t.shape for t in targets_list]}')  # Debug print

            # Ensure inputs and targets are Tensors and move them to the correct device
            inputs = inputs.to(device)
            targets_list = [t.to(device) for t in targets_list]

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # Generator forward pass with cached activations
            outputs = model(inputs)

            # Calculate loss, one stem at a time
            loss_g_total = 0.0
            for stem, target_stem in enumerate(targets_list):
                output_stem = outputs[:, stem]
                loss_g_stem = criterion(output_stem, target_stem)
                loss_g_total += loss_g_stem

            loss_g_total.backward()  # Backpropagate on the total generator loss
            optimizer_g.step()
            running_loss_g += loss_g_total.item()

            # Discriminator forward pass
            real_labels = torch.ones(targets_list[0].size(0), 1, device=device)
            fake_labels = torch.zeros(outputs.size(0), 1, device=device)

            for stem, target_stem in enumerate(targets_list):
                with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temp_outputs_file:
                    outputs_np = np.memmap(temp_outputs_file.name, dtype=np.float32, mode='w+', shape=tuple(outputs[:, stem].shape))
                    outputs_np[:] = outputs[:, stem].detach().cpu().numpy()[:]

                real_out = discriminator(target_stem.unsqueeze(1))
                fake_out = discriminator(torch.from_numpy(outputs_np).to(device, non_blocking=True).unsqueeze(1).detach())
                loss_d_real = criterion(real_out, real_labels)
                loss_d_fake = criterion(fake_out, fake_labels)

                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                optimizer_d.step()
                running_loss_d += loss_d.item()

                # Clean up temporary files
                del outputs_np
                os.remove(temp_outputs_file.name)

            if i % accumulation_steps == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                running_loss_g = 0.0
                running_loss_d = 0.0

            # Explicitly run garbage collection and clear CUDA cache at the end of each iteration
            del outputs, targets_list  # Delete tensors from GPU
            gc.collect()  # Explicitly run garbage collection
            torch.cuda.empty_cache()  # Clear the CUDA cache

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

    final_model_path = f"{checkpoint_dir}/model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved at {final_model_path}")

def start_training(data_dir, batch_size, num_epochs, learning_rate, use_cuda, checkpoint_dir, save_interval, accumulation_steps, num_stems, num_workers, cache_dir):
    logger.info(f"Starting training with dataset at {data_dir}")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sample_rate, n_mels, n_fft = detect_parameters(data_dir)
    target_length = 256  # Define your target length here

    dataset = StemSeparationDataset(data_dir, n_mels=n_mels, target_length=target_length, n_fft=n_fft, num_stems=num_stems, cache_dir=cache_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if use_cuda else False, num_workers=num_workers)

    model = KANWithDepthwiseConv(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length, num_stems=num_stems, cache_dir=cache_dir).to(device)
    discriminator = KANDiscriminator(in_channels=1, out_channels=64, n_mels=n_mels, target_length=target_length).to(device)
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets_list = data
            logger.debug(f"Type of inputs: {type(inputs)}, Type of targets_list: {type(targets_list)}")
            print(f'Batch {i} inputs shape: {inputs.shape}')  # Debug print
            print(f'Batch {i} targets shape: {[t.shape for t in targets_list]}')  # Debug print

            # Ensure inputs and targets are Tensors and move them to the correct device
            inputs = inputs.to(device)
            targets_list = [t.to(device) for t in targets_list]

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # Generator forward pass with cached activations
            outputs = model(inputs)

            # Calculate loss, one stem at a time
            loss_g_total = 0.0
            for stem, target_stem in enumerate(targets_list):
                output_stem = outputs[:, stem]
                loss_g_stem = criterion(output_stem, target_stem)
                loss_g_total += loss_g_stem

            loss_g_total.backward()  # Backpropagate on the total generator loss
            optimizer_g.step()
            running_loss_g += loss_g_total.item()

            # Discriminator forward pass
            real_labels = torch.ones(targets_list[0].size(0), 1, device=device)
            fake_labels = torch.zeros(outputs.size(0), 1, device=device)

            for stem, target_stem in enumerate(targets_list):
                with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temp_outputs_file:
                    outputs_np = np.memmap(temp_outputs_file.name, dtype=np.float32, mode='w+', shape=tuple(outputs[:, stem].shape))
                    outputs_np[:] = outputs[:, stem].detach().cpu().numpy()[:]

                real_out = discriminator(target_stem.unsqueeze(1))
                fake_out = discriminator(torch.from_numpy(outputs_np).to(device, non_blocking=True).unsqueeze(1).detach())
                loss_d_real = criterion(real_out, real_labels)
                loss_d_fake = criterion(fake_out, fake_labels)

                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                optimizer_d.step()
                running_loss_d += loss_d.item()

                # Clean up temporary files
                del outputs_np
                os.remove(temp_outputs_file.name)

            if i % accumulation_steps == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss G: {running_loss_g/accumulation_steps:.4f}, Loss D: {running_loss_d/accumulation_steps:.4f}')
                running_loss_g = 0.0
                running_loss_d = 0.0

            # Explicitly run garbage collection and clear CUDA cache at the end of each iteration
            del outputs, targets_list  # Delete tensors from GPU
            gc.collect()  # Explicitly run garbage collection
            torch.cuda.empty_cache()  # Clear the CUDA cache

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

    final_model_path = f"{checkpoint_dir}/model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved at {final_model_path}")
