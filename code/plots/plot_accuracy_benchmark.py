from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# Project paths
ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "results" / "benchmarks"
FIGURES = ROOT / "results" / "figures" / "benchmarks"
FIGURES.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = BENCHMARKS / "accuracy_summary.csv"
PREDICTIONS_PATH = BENCHMARKS / "accuracy_predictions.csv"


# Consistent appearance across the accuracy figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})

METHODS = {
    "Monte Carlo (5000 paths)": {
        "label": "Monte Carlo (5,000 paths)",
        "error": "mc_error",
        "color": "tab:red",
    },
    "ANN-MC": {
        "label": "ANN trained on MC",
        "error": "ann_mc_error",
        "color": "tab:blue",
    },
    "ANN-Analytical": {
        "label": "ANN trained on analytical prices",
        "error": "ann_analytical_error",
        "color": "tab:green",
    },
}


def save_plot(filename):
    path = FIGURES / filename
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# Load results from the held-out test set.
summary = pd.read_csv(SUMMARY_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

required_summary = {"Method", "MAE", "RMSE"}
required_predictions = {"lambda", "mc_error", "ann_mc_error", "ann_analytical_error"}

if not required_summary.issubset(summary.columns):
    raise ValueError(f"Missing columns in {SUMMARY_PATH.name}: {required_summary - set(summary.columns)}")
if not required_predictions.issubset(predictions.columns):
    raise ValueError(f"Missing columns in {PREDICTIONS_PATH.name}: {required_predictions - set(predictions.columns)}")
if summary["Method"].duplicated().any():
    raise ValueError("Duplicate methods in accuracy_summary.csv")
if not np.isfinite(predictions[list(required_predictions)].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values in accuracy_predictions.csv")

summary = summary.set_index("Method")
missing_methods = set(METHODS) - set(summary.index)
if missing_methods:
    raise ValueError(f"Missing methods in summary CSV: {sorted(missing_methods)}")

# Check that the summary and individual predictions describe the same test run.
for method, settings in METHODS.items():
    errors = predictions[settings["error"]].to_numpy(dtype=float)
    actual_mae = np.mean(np.abs(errors))
    actual_rmse = np.sqrt(np.mean(errors ** 2))
    if not np.isclose(actual_mae, float(summary.loc[method, "MAE"]), atol=1e-3, rtol=0):
        raise ValueError(f"MAE mismatch for {method}: {actual_mae:.6f} vs summary")
    if not np.isclose(actual_rmse, float(summary.loc[method, "RMSE"]), atol=1e-3, rtol=0):
        raise ValueError(f"RMSE mismatch for {method}: {actual_rmse:.6f} vs summary")

print(f"Test options: {len(predictions):,}")


# --------------------------------------------------
# Plot 1: MAE and RMSE, both in option-price units.
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))
metrics = ["MAE", "RMSE"]
x = np.arange(len(metrics))
width = 0.24

for i, (method, settings) in enumerate(METHODS.items()):
    values = summary.loc[method, metrics].to_numpy(dtype=float)
    offsets = x + (i - 1) * width
    bars = ax.bar(offsets, values, width=width,
                  label=settings["label"], color=settings["color"])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

ax.set_xticks(x, metrics)
ax.set_xlabel("Error metric")
ax.set_ylabel("Pricing error metric (price units)")
ax.set_ylim(0, summary.loc[list(METHODS), metrics].to_numpy().max() * 1.2)
ax.grid(axis="y", alpha=0.22)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)
save_plot("accuracy_comparison.pdf")


# --------------------------------------------------
# Plot 2: Cumulative distribution of absolute errors.
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

# Focus on a common range containing at least 99% of each method's errors.
upper = max(
    np.quantile(np.abs(predictions[s["error"]].to_numpy(dtype=float)), 0.99)
    for s in METHODS.values()
)
upper = np.ceil(upper * 10) / 10

for method, settings in METHODS.items():
    errors = np.sort(np.abs(predictions[settings["error"]].to_numpy(dtype=float)))
    n = len(errors)
    # Plot at most 2500 ordered points to keep the vector PDF compact.
    idx = np.unique(np.linspace(0, n - 1, min(n, 2500), dtype=int))
    ax.plot(errors[idx], (idx + 1) / n,
            label=settings["label"], color=settings["color"], linewidth=1.9)

ax.set_xlim(0, upper)
ax.set_ylim(0, 1.01)
ax.set_xlabel("Absolute pricing error (price units)")
ax.set_ylabel("Options with error at or below threshold")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.grid(alpha=0.22)
ax.legend(loc="lower right", frameon=False)
save_plot("error_distribution.pdf")


# --------------------------------------------------
# Plot 3: MAE in bins of jump intensity (lambda).
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

lam = predictions["lambda"].to_numpy(dtype=float)
bin_edges = np.linspace(0, 1.5, 11)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2

for method, settings in METHODS.items():
    errors = np.abs(predictions[settings["error"]].to_numpy(dtype=float))
    mae_per_bin = []

    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Include the upper edge only in the last interval.
        mask = (lam >= low) & ((lam <= high) if i == len(centers) - 1 else (lam < high))
        if not mask.any():
            raise ValueError(f"Empty lambda bin [{low:.2f}, {high:.2f}]")
        mae_per_bin.append(errors[mask].mean())

    ax.plot(centers, mae_per_bin, "o-", markersize=4.5, linewidth=1.8,
            label=settings["label"], color=settings["color"])

ax.set_xlim(0, 1.5)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"Jump intensity $\lambda$")
ax.set_ylabel("Mean absolute pricing error (price units)")
ax.grid(alpha=0.22)
ax.legend(frameon=False)
save_plot("accuracy_vs_lambda.pdf")

print("All accuracy benchmark plots generated successfully.")
