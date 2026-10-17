from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"


def main():
    # The Monte Carlo prices already exist. I just need the same split.
    df = pd.read_csv(DATA / "merton_dataset2.csv")

    y = df["price"].to_numpy(dtype=np.float64).reshape(-1, 1)

    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Both networks use the same inputs, but need their own output scaler.
    scaler = StandardScaler()

    y_train = scaler.fit_transform(y_train)
    y_val = scaler.transform(y_val)
    y_test = scaler.transform(y_test)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED / "y_train_mc_comparison.npy", y_train)
    np.save(PROCESSED / "y_val_mc_comparison.npy", y_val)
    np.save(PROCESSED / "y_test_mc_comparison.npy", y_test)

    joblib.dump(scaler, PROCESSED / "y_scaler_mc_comparison.pkl")

    print("Monte Carlo comparison dataset prepared.")
    print("Train:", y_train.shape)
    print("Validation:", y_val.shape)
    print("Test:", y_test.shape)

    print("\nOutput scaler:")
    print("Mean:", scaler.mean_[0])
    print("Scale:", scaler.scale_[0])


if __name__ == "__main__":
    main()