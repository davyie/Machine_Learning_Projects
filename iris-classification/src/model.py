from torch import nn
import torch

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        logits = self.linear_relu(x)
        return logits
    
    