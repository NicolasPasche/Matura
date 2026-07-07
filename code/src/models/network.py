import torch
import torch.nn as nn

class PricingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 128), 
            nn.ReLU(), 

            nn.Linear(128, 128), 
            nn.ReLU(), 

            nn.Linear(128, 64), 
            nn.ReLU(), 

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)