import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_channel=768, hidden_layer=256):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
