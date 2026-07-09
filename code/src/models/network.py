import torch
import torch.nn as nn

class PricingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(9, 128), 
            nn.ReLU(), 

            nn.Linear(128, 128), 
            nn.ReLU(), 

            nn.Linear(128, 64), 
            nn.ReLU(), 

            nn.Linear(64, 1) 
            # The model was returning negative prices,
            # So i added Softplus to force positive outputs
            # nn.Softplus()
            
            # Well... That didn't work, the model didn't learn anything
            # I'll just use np.maximum() when evaluating
        )

    def forward(self, x):
        return self.network(x)