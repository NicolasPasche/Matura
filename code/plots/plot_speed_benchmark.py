from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "benchmarks" / "speed_benchmark.csv"
OUTPUT_DIR = ROOT / "results" / "figures" / "benchmarks"

COLORS = {
    "Analytical CPU": "black",
    "Monte Carlo CPU": "tab:red",
    "ANN CPU Batch": "tab:blue",
    "ANN CPU Single": "tab:cyan",
    "ANN GPU Batch": "tab:green",
    "ANN GPU Single": "tab:orange",
}

MARKERS = {
    "Analytical CPU": "o",
    "Monte Carlo CPU": "s",
    "ANN CPU Batch": "^",
    "ANN CPU Single": "v",
    "ANN GPU Batch": "D",
    "ANN GPU Single": "x",
}


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })

    df = pd.read_csv(DATA_PATH)
    required = {"N", "method", "median_ms", "options_per_second"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.duplicated(["N", "method"]).any():
        raise ValueError("Duplicate (N, method) entries in benchmark CSV.")
    if (df["median_ms"] <= 0).any() or (df["N"] <= 0).any():
        raise ValueError("Batch sizes and execution times must be positive.")
    df = df.sort_values("N")

    # 1. Execution time
    plt.figure()
    for method, color in COLORS.items():
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        plt.plot(
            subset["N"], subset["median_ms"],
            label=method, color=color, marker=MARKERS[method],
            markersize=4, linewidth=1.5,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of options")
    plt.ylabel("Execution time (ms)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    save_plot("pricing_time.pdf")

    # 2. GPU speedup relative to analytical CPU pricing
    analytical = df[df["method"] == "Analytical CPU"][["N", "median_ms"]]
    gpu = df[df["method"] == "ANN GPU Batch"][["N", "median_ms"]]
    comparison = analytical.merge(gpu, on="N", suffixes=("_analytical", "_gpu"))
    if comparison.empty:
        raise ValueError("No matching analytical CPU and ANN GPU batch results.")
    comparison["speedup"] = (
        comparison["median_ms_analytical"] / comparison["median_ms_gpu"]
    )

    plt.figure()
    plt.plot(
        comparison["N"], comparison["speedup"],
        color=COLORS["ANN GPU Batch"], marker="o", markersize=4,
        linewidth=1.6, label="Analytical CPU / ANN GPU Batch",
    )
    plt.axhline(1, color="black", linestyle="--", linewidth=1,
                label="Equal execution time")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of options")
    plt.ylabel("Speedup")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    save_plot("speedup.pdf")

    # 3. Throughput
    plt.figure()
    for method in ("Analytical CPU", "ANN CPU Batch", "ANN GPU Batch"):
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        plt.plot(
            subset["N"], subset["options_per_second"] / 1e6,
            label=method, color=COLORS[method], marker=MARKERS[method],
            markersize=4, linewidth=1.5,
        )
    plt.xscale("log")
    plt.xlabel("Number of options")
    plt.ylabel("Throughput (million options/s)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    save_plot("throughput.pdf")

    best = comparison.loc[comparison["speedup"].idxmax()]
    faster = comparison[comparison["speedup"] > 1]
    print(f"Maximum measured speedup: {best['speedup']:.2f}x "
          f"at N={int(best['N']):,}")
    if not faster.empty:
        print(f"First measured faster GPU batch: N={int(faster['N'].min()):,}")


if __name__ == "__main__":
    main()
