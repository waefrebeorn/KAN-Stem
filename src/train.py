import torch
from dataset import MyAudioDataset
from KANModel import KANModel
from torch.utils.data import DataLoader
import torch.optim as optim
from losses import sisnr_loss

def train_model():
    train_dataset = MyAudioDataset(r'G:\Music\badmultitracks-michaeljackson\dataset\wav')  # Replace with actual path
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Number of epochs
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = sisnr_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_model()
