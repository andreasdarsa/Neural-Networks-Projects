# This serves as a representation of a feedforward neural network (FNN) using PyTorch.
# The network will use MLP architecture
import torch.nn as nn

class MLPFeedforwardNN(nn.Module):
    def __init__(self, in_dim=10, hidden=64, out_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.layers(x)