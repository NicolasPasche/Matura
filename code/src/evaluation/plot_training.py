import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "results" / "figures" / "ann_analytical"

sys.path.append(str(ROOT / "code" / "src"))
from models.network import PricingNetwork

FIGURES.mkdir(parents=True, exist_ok=True)


# Load the training history
history = np.load(MODELS / "training_history_analytical.npz")

train_loss = history["train_loss"]
val_loss = history["val_loss"]
epochs = np.arange(1, len(train_loss) + 1)


# Plot 1: Learning curves
plt.figure(figsize=(9, 5))

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (log scale)")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "learning_curves_log.png", dpi=300)
plt.close()


# Load the best model for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PricingNetwork().to(device)
model.load_state_dict(
    torch.load(
        MODELS / "best_model_analytical.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

X_val = np.load(DATA / "X_val_comparison.npy")
y_val = np.load(DATA / "y_val_analytical.npy")

y_scaler = joblib.load(DATA / "y_scaler_analytical.pkl")


# Evaluate in batches to avoid unnecessary memory usage.
predictions = []

with torch.no_grad():
    for batch in np.array_split(X_val, max(1, int(np.ceil(len(X_val) / 8192)))):
        x = torch.tensor(batch, dtype=torch.float32, device=device)
        predictions.append(model(x).cpu().numpy())

predictions = np.vstack(predictions)

# The model predicts standardized prices, so I need to undo the scaling.
y_pred = y_scaler.inverse_transform(predictions).ravel()
y_true = y_scaler.inverse_transform(y_val).ravel()

errors = y_pred - y_true

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))

print(f"Validation MAE:  {mae:.6f}")
print(f"Validation RMSE: {rmse:.6f}")


# Plot 2: Predicted vs. analytical prices
# A perfect model would place every point on the diagonal.
rng = np.random.default_rng(42)
indices = rng.choice(len(y_true), size=min(5000, len(y_true)), replace=False)

plt.figure(figsize=(7, 7))

plt.scatter(y_true[indices], y_pred[indices], s=5, alpha=0.25)

low = min(y_true.min(), y_pred.min())
high = max(y_true.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--", label="Perfect prediction")

plt.xlabel("Analytical Merton Price")
plt.ylabel("ANN Prediction")
plt.title("Predicted vs. Analytical Prices")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "predicted_vs_analytical.png", dpi=300)
plt.close()


# Plot 3: Error distribution
# I want to see whether the network makes mostly small errors
# or whether a few bad predictions dominate the results.
plt.figure(figsize=(9, 5))

plt.hist(errors, bins=100, edgecolor="none", alpha=0.8)
plt.axvline(0, color="red", linestyle="--", label="Zero error")

plt.xlabel("Prediction Error (ANN - Analytical)")
plt.ylabel("Number of Options")
plt.title("ANN Prediction Error Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "error_distribution.png", dpi=300)
plt.close()


print(f"\nPlots saved to: {FIGURES}")


# I want to understand which options are the hardest to price.
worst_indices = np.argsort(np.abs(errors))[-5:][::-1]

x_scaler = joblib.load(DATA / "x_scaler_comparison.pkl")
X_original = x_scaler.inverse_transform(X_val)

features = ["S0", "K", "T", "r", "d", "sigma", "lambda", "muJ", "sigmaJ"]

print("\nFive largest prediction errors:\n")

for i in worst_indices:
    print(f"Analytical: {y_true[i]:.6f}")
    print(f"ANN:        {y_pred[i]:.6f}")
    print(f"Error:      {errors[i]:+.6f}")

    for name, value in zip(features, X_original[i]):
        print(f"  {name}: {value:.6f}")

    print()


# I want to check whether the largest errors are just random outliers
# or whether certain parameter ranges are genuinely harder to price.

def analyze_parameter(name, values, bins):
    print(f"\nError analysis for {name}:")
    print(f"{'Range':<18} {'Options':>10} {'MAE':>12} {'RMSE':>12}")

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (values >= low) & (values < high)

        if not np.any(mask):
            continue

        subset = errors[mask]

        mae = np.mean(np.abs(subset))
        rmse = np.sqrt(np.mean(subset ** 2))

        print(
            f"{low:.2f} - {high:.2f}   "
            f"{len(subset):>10} "
            f"{mae:>12.6f} "
            f"{rmse:>12.6f}"
        )


x_scaler = joblib.load(DATA / "x_scaler_comparison.pkl")
X_original = x_scaler.inverse_transform(X_val)

sigma = X_original[:, 5]
lam = X_original[:, 6]
sigma_j = X_original[:, 8]

analyze_parameter(
    "sigma",
    sigma,
    [0.05, 0.1, 0.2, 0.3, 0.4, 0.500001]
)

analyze_parameter(
    "lambda",
    lam,
    [0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.500001]
)

analyze_parameter(
    "sigmaJ",
    sigma_j,
    [0.05, 0.2, 0.35, 0.45, 0.500001]
)