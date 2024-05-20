import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.poly = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.relu(self.linear(x) + self.poly(x ** 2))

class KANModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KANModel, self).__init__()
        self.layer1 = KANLayer(input_size, hidden_size)
        self.layer2 = KANLayer(hidden_size, hidden_size)
        self.layer3 = KANLayer(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)
