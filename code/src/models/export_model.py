import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]

sys.path.append(str(project_root / "code" / "src"))

from models.network import PricingNetwork


model_path = project_root / "models" /"best_model_dividend.pth"
output_path = project_root / "models" / "best_model_dividend.pt"


model = PricingNetwork()

state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

model.load_state_dict(state_dict)
model.eval()



example_input = torch.randn(1, 9)

scripted_model = torch.jit.trace(model, example_input)

scripted_model.save(output_path)

print(f"Model exported to: {output_path}")