import numpy as np
import torch
import joblib

from pathlib import Path
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Project paths

project_root = Path(__file__).parent.parent.parent.parent

src_path = project_root / "code" / "src"

sys.path.append(str(src_path))

from models.network import PricingNetwork


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Load data and trained model

data_path = project_root / "data" / "processed"
model_path = project_root / "models" / "best_model.pth"


X_test = np.load(data_path / "X_test.npy")

y_test = np.load(data_path / "y_test.npy")


print("\nTest data:")
print(X_test.shape)
print(y_test.shape)


# Load scalers
y_scaler = joblib.load(data_path / "y_scaler.pkl")

# Convert data to tensors
X_test_tensor = torch.tensor(
    X_test, 
    dtype=torch.float32
).to(device)

# Load trained model
model = PricingNetwork().to(device)

model.load_state_dict(
    torch.load(
        model_path, 
        map_location=device, 
        weights_only=True
    )
)

model.eval()

print("\nModel loaded.")


# Prediction
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)

y_pred_scaled = y_pred_scaled.cpu().numpy()


# Transform back to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)

y_true = y_scaler.inverse_transform(y_test)


# Evaluation metrics
mse = mean_squared_error(y_true, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true, y_pred)

r2 = r2_score(y_true, y_pred)


print("\nMODEL EVALUATION")

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

print("\nExample predictions:")

for i in range(10):

    print(
        f"True: {y_true[i][0]:8.4f} "
        f"Predicted: {y_pred[i][0]:8.4f} "
        f"Error: {(y_pred[i][0]-y_true[i][0]):8.4f}"
    )