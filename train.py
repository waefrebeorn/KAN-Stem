import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import KAN
from data_loader import get_data_loader

def train_model(dataset_path, num_epochs=10, batch_size=16, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KAN().to(device)
    data_loader = get_data_loader(dataset_path, batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, sample_rate) in enumerate(data_loader, 0):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                writer.add_scalar('training_loss', running_loss / 10, epoch * len(data_loader) + i)
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), 'kan_model.pth')
    writer.close()
