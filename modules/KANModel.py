import torch
import torch.nn as nn

class KANModel(nn.Module):
    def __init__(self):
        super(KANModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 128, 1024)
        self.fc2 = nn.Linear(1024, 4 * 44100)  # Assuming output size matches the target shape

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 4, 44100)  # Reshape to match target shape
        return x
