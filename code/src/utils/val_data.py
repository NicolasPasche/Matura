import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# Style (paper quality light)
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv("../data/merton_dataset.csv")

print("\n==================== FIRST 5 ROWS ====================\n")
print(df.head())

print("\n==================== DATASET SUMMARY ==================\n")
print(df.describe())

print("\n==================== MISSING VALUES ===================\n")
print(df.isnull().sum())

print("\n==================== DUPLICATES =======================\n")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n==================== PRICE STATISTICS =================\n")
print(df["price"].describe())

print("\n==================== CORRELATION MATRIX ===============\n")
print(df.corr(numeric_only=True))

# ============================================================
# Downsample for scatter / 3D
# ============================================================

df_small = df.sample(20000, random_state=42)

# ============================================================
# Histogram
# ============================================================

plt.figure(figsize=(7,4))
plt.hist(df["price"], bins=120)
plt.yscale("log")
plt.title("Distribution of Option Prices")
plt.xlabel("Price")
plt.ylabel("Frequency (log scale)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ============================================================
# 3D Plot 1: S0 vs K vs Price
# ============================================================

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_small["S0"], df_small["K"], df_small["price"], s=1, alpha=0.25)

ax.set_title("Option Price Surface (S0 vs K)")
ax.set_xlabel("S0")
ax.set_ylabel("K")
ax.set_zlabel("Price")

plt.tight_layout()
plt.show()

# ============================================================
# FIXED HEATMAP (NO PIVOT -> HEXBIN)
# ============================================================

plt.figure(figsize=(7,5))

hb = plt.hexbin(
    df["S0"],
    df["K"],
    C=np.log1p(df["price"]),   # important: stabilizes scale
    gridsize=70,
    reduce_C_function=np.mean,
    cmap="viridis",
    mincnt=1
)

plt.colorbar(hb, label="log(1 + Price)")
plt.title("Option Price Surface (S0 vs K)")
plt.xlabel("S0")
plt.ylabel("K")

plt.tight_layout()
plt.show()

# ============================================================
# 3D Plot 2: S0 vs sigma vs Price
# ============================================================

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_small["S0"], df_small["sigma"], df_small["price"], s=1, alpha=0.25)

ax.set_title("Option Price (S0 vs Volatility)")
ax.set_xlabel("S0")
ax.set_ylabel("sigma")
ax.set_zlabel("Price")

plt.tight_layout()
plt.show()

# ============================================================
# 3D Plot 3: Moneyness vs T vs Price
# ============================================================

moneyness = df_small["S0"] / df_small["K"]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(moneyness, df_small["T"], df_small["price"], s=1, alpha=0.25)

ax.set_title("Option Price (Moneyness vs Time)")
ax.set_xlabel("S0/K")
ax.set_ylabel("T")
ax.set_zlabel("Price")

plt.tight_layout()
plt.show()

# ============================================================
# 3D Plot 4: Merton jumps
# ============================================================

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_small["lambda"], df_small["muJ"], df_small["price"], s=1, alpha=0.25)

ax.set_title("Merton Jump Effects")
ax.set_xlabel("lambda")
ax.set_ylabel("muJ")
ax.set_zlabel("Price")

plt.tight_layout()
plt.show()

# ============================================================
# 2D sanity checks (unchanged)
# ============================================================

plt.figure(figsize=(7,4))
plt.scatter(df["S0"], df["price"], s=1, alpha=0.2)
plt.title("S0 vs Price")
plt.xlabel("S0")
plt.ylabel("Price")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.scatter(df["K"], df["price"], s=1, alpha=0.2)
plt.title("K vs Price")
plt.xlabel("K")
plt.ylabel("Price")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()