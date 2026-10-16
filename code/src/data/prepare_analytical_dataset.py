# FOR USERS:
# The dataset "merton_dataset2.csv" has the exact same features as merton_dataset_analytical.csv
# but the target has been calculated using Monte Carlo simulation, it was my mistake.
# So in merton_dataset_analytical.csv, the target was calculated using the analytical formula.
# In this script, we are just switching around the target values, to save time when preparing the analytical dataset.

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MC_DATASET = DATA_DIR / "merton_dataset2.csv"
ANALYTICAL_DATASET = DATA_DIR / "merton_dataset_analytical.csv"


FEATURES = [
    "S0",
    "K",
    "T",
    "r",
    "d",
    "sigma",
    "lambda",
    "muJ",
    "sigmaJ"
]


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    mc_df = pd.read_csv(MC_DATASET)
    analytical_df = pd.read_csv(ANALYTICAL_DATASET)

    print("Monte Carlo dataset:", mc_df.shape)
    print("Analytical dataset:", analytical_df.shape)

    # To compare the network trained on mc data and on analytical data, we need to 
    # guarantee that the features (input parameters) are the same in both datasets.
    # Here, we check if the features match. If not, an error is raised.
    if len(mc_df) != len(analytical_df):
        raise ValueError("Dataset sizes do not match.")

    X_mc = mc_df[FEATURES].to_numpy(dtype=np.float64)
    X_analytical = analytical_df[FEATURES].to_numpy(dtype=np.float64)

    if not np.array_equal(X_mc, X_analytical):
        raise ValueError("Input parameters do not match.")

    if not np.all(np.isfinite(X_mc)):
        raise ValueError("Input parameters contain invalid values.")

    print("Features match.")


    # Split row indices so both targets use exactly the same samples.
    indices = np.arange(len(mc_df))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size= 0.3,
        random_state = 42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42
    )

    print("\nDataset split:")
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))


    # Fit the input scaler only on training samples.
    # Both networks will use this same scaler.
    x_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_mc[train_idx])
    X_val = x_scaler.transform(X_mc[val_idx])
    X_test = x_scaler.transform(X_mc[test_idx])


    # The analytical model needs its own target scaler.
    y = analytical_df["price"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(y)):
        raise ValueError("Analytical prices contain invalid values.")

    if np.any(y < 0):
        raise ValueError("Analytical prices contain negative values.")

    y_train = y[train_idx].reshape(-1, 1)
    y_val = y[val_idx].reshape(-1, 1)
    y_test = y[test_idx].reshape(-1,1)

    y_scaler = StandardScaler()

    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)


    # Separate filenames protect the existing trained model's data.
    np.save(PROCESSED_DIR / "X_train_comparison.npy", X_train)
    np.save(PROCESSED_DIR / "X_val_comparison.npy", X_val)
    np.save(PROCESSED_DIR / "X_test_comparison.npy", X_test)
    np.save(PROCESSED_DIR / "y_train_analytical.npy", y_train)
    np.save(PROCESSED_DIR / "y_val_analytical.npy", y_val)
    np.save(PROCESSED_DIR / "y_test_analytical.npy",y_test)

    joblib.dump(x_scaler, PROCESSED_DIR / "x_scaler_comparison.pkl")
    joblib.dump(y_scaler, PROCESSED_DIR / "y_scaler_analytical.pkl")

    print("\nInput shape:", X_train.shape)

    print("\nInput scaler:")
    print("Mean:", x_scaler.mean_)
    print("Scale:", x_scaler.scale_)

    print("\nTarget scaler:")
    print("Mean:", y_scaler.mean_[0])
    print("Scale:", y_scaler.scale_[0])

    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()