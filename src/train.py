import os
import torch
from torch.utils.data import DataLoader
from modules.KANModel import KANModel  # Corrected import path
from dataset import MyAudioDataset  # Ensure this path is correct based on your structure
from losses import si_snr_loss  # Ensure this path is correct based on your structure

def train_model():
    # Initialize dataset and dataloader
    train_dataset = MyAudioDataset('G:/Music/badmultitracks-michaeljackson/dataset/wav')  # Update with your actual dataset path
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)  # Reduced batch size and num_workers

    # Initialize model, loss function, and optimizer
    model = KANModel().cuda()  # Ensure the model is on the GPU
    criterion = si_snr_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    accumulation_steps = 4  # Number of steps to accumulate gradients

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

if __name__ == "__main__":
    train_model()
