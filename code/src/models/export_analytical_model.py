import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "code" / "src"))

from models.network import PricingNetwork


# Load the trained PyTorch model.
model = PricingNetwork()

model.load_state_dict(
    torch.load(
        ROOT / "models" / "best_model_analytical.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# C++ needs TorchScript, not the original PyTorch checkpoint.
example_input = torch.randn(1, 9)

scripted_model = torch.jit.trace(model, example_input)
scripted_model.save(str(ROOT / "models" / "best_model_analytical.pt"))

print("Analytical ANN exported successfully.")