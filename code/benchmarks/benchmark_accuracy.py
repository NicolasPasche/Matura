import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "benchmarks"

sys.path.append(str(ROOT / "code" / "src"))
from models.network import PricingNetwork


RESULTS.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8192
FEATURES = ["S0", "K", "T", "r", "d", "sigma", "lambda", "muJ", "sigmaJ"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def predict(model_path, X):
    model = PricingNetwork().to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # I evaluate both networks in batches to keep memory usage reasonable.
    predictions = []

    with torch.inference_mode():
        for i in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[i:i+BATCH_SIZE]).to(device)
            predictions.append(model(batch).cpu().numpy())

    return np.concatenate(predictions).ravel()


def calculate_metrics(predictions, reference):
    errors = predictions - reference
    absolute = np.abs(errors)

    return {
        "MAE": np.mean(absolute),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "Bias": np.mean(errors),
        "P95": np.percentile(absolute, 95),
        "P99": np.percentile(absolute, 99),
        "MaxAE": np.max(absolute)
    }


def main():
    # All methods are evaluated against the same analytical reference prices.
    X = np.load(DATA / "X_test_comparison.npy").astype(np.float32)

    y_analytical = np.load(DATA / "y_test_analytical.npy")
    y_mc = np.load(DATA / "y_test_mc_comparison.npy")

    x_scaler = joblib.load(DATA / "x_scaler_comparison.pkl")
    analytical_scaler = joblib.load(DATA / "y_scaler_analytical.pkl")
    mc_scaler = joblib.load(DATA / "y_scaler_mc_comparison.pkl")

    reference = analytical_scaler.inverse_transform(y_analytical).ravel()
    mc_prices = mc_scaler.inverse_transform(y_mc).ravel()

    print(f"\nTest options: {len(X):,}")

    # Each model needs its own output scaler.
    mc_scaled = predict(MODELS / "best_model_mc_comparison.pth", X)
    analytical_scaled = predict(MODELS / "best_model_analytical.pth", X)

    mc_ann = mc_scaled * mc_scaler.scale_[0] + mc_scaler.mean_[0]
    analytical_ann = (
        analytical_scaled * analytical_scaler.scale_[0]
        + analytical_scaler.mean_[0]
    )

    predictions = {
        "Monte Carlo (5000 paths)": mc_prices,
        "ANN-MC": mc_ann,
        "ANN-Analytical": analytical_ann
    }

    # The Monte Carlo estimates provide a useful noise baseline.
    results = []

    print("\nAccuracy benchmark")
    print("-" * 85)
    print(
        f"{'Method':<28} {'MAE':>10} {'RMSE':>10} "
        f"{'P95':>10} {'P99':>10} {'MaxAE':>10}"
    )

    for name, values in predictions.items():
        metrics = calculate_metrics(values, reference)
        results.append({"Method": name, **metrics})

        print(
            f"{name:<28} "
            f"{metrics['MAE']:>10.6f} "
            f"{metrics['RMSE']:>10.6f} "
            f"{metrics['P95']:>10.6f} "
            f"{metrics['P99']:>10.6f} "
            f"{metrics['MaxAE']:>10.6f}"
        )

    pd.DataFrame(results).to_csv(
        RESULTS / "accuracy_summary.csv", index=False
    )

    # Save individual predictions so I can analyze errors without running
    # both networks again.
    original_X = x_scaler.inverse_transform(X)

    df = pd.DataFrame(original_X, columns=FEATURES)

    df["analytical_price"] = reference
    df["mc_price"] = mc_prices
    df["ann_mc_price"] = mc_ann
    df["ann_analytical_price"] = analytical_ann

    for name, values in [
        ("mc", mc_prices),
        ("ann_mc", mc_ann),
        ("ann_analytical", analytical_ann)
    ]:
        df[f"{name}_error"] = values - reference

    df.to_csv(RESULTS / "accuracy_predictions.csv", index=False)

    # Compare the absolute errors of both ANNs.
    plt.figure(figsize=(9, 5))

    errors_mc = np.abs(mc_ann - reference)
    errors_analytical = np.abs(analytical_ann - reference)

    # Zoom into the main distribution without changing the reported metrics.
    limit = np.percentile(
        np.concatenate([errors_mc, errors_analytical]), 99
    )

    bins = np.linspace(0, limit, 80)

    plt.hist(errors_mc, bins=bins, alpha=0.55, label="ANN-MC")
    plt.hist(errors_analytical, bins=bins, alpha=0.55, label="ANN-Analytical")

    plt.xlabel("Absolute Pricing Error")
    plt.ylabel("Number of Options")
    plt.title("ANN Accuracy Comparison (Zoomed)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_distribution.png", dpi=300)
    plt.close()

    # I want to see whether both networks struggle with rare jumps.
    lambdas = original_X[:, 6]
    lambda_bins = np.linspace(0, 1.5, 16)
    centers = (lambda_bins[:-1] + lambda_bins[1:]) / 2

    plt.figure(figsize=(9, 5))

    for name, errors in [
        ("ANN-MC", errors_mc),
        ("ANN-Analytical", errors_analytical)
    ]:
        bin_mae = []

        for i, (low, high) in enumerate(zip(lambda_bins[:-1], lambda_bins[1:])):
            mask = (lambdas >= low) & (
                lambdas <= high if i == len(centers) - 1 else lambdas < high
            )
            bin_mae.append(np.mean(errors[mask]) if np.any(mask) else np.nan)

        plt.plot(centers, bin_mae, marker="o", label=name)

    plt.xlabel("Jump Intensity (lambda)")
    plt.ylabel("Mean Absolute Error")
    plt.title("ANN Accuracy vs. Jump Intensity")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_lambda.png", dpi=300)
    plt.close()

    print(f"\nResults saved to: {RESULTS}")


if __name__ == "__main__":
    main()