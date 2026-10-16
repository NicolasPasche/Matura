import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

sys.path.append(str(PROJECT_ROOT / "code" / "src"))
from models.network import PricingNetwork


# Training parameters
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 42


def set_seed(seed):
    # I want both networks to start under the same conditions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def load_data(dataset_type):
    # Both networks get the same inputs. Only the target prices change.
    X_train = np.load(DATA_PATH / "X_train_comparison.npy")
    X_val = np.load(DATA_PATH / "X_val_comparison.npy")

    suffix = "analytical" if dataset_type == "analytical" else "mc_comparison"

    y_train = np.load(DATA_PATH / f"y_train_{suffix}.npy")
    y_val = np.load(DATA_PATH / f"y_val_{suffix}.npy")

    print(f"\nTraining data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")

    if X_train.shape[1] != 9 or X_val.shape[1] != 9:
        raise ValueError("Expected 9 input features.")

    if y_train.shape != (len(X_train), 1) or y_val.shape != (len(X_val), 1):
        raise ValueError("Invalid target dimensions.")

    arrays = [X_train, X_val, y_train, y_val]

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Dataset contains invalid values.")

    return [
        torch.tensor(array, dtype=torch.float32)
        for array in arrays
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mc", "analytical"], required=True)
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    filename = (
        "best_model_analytical.pth"
        if args.dataset == "analytical"
        else "best_model_mc_comparison.pth"
    )
    save_path = MODEL_PATH / filename

    # I really don't want to overwrite my original trained model.
    if save_path.exists():
        raise FileExistsError(f"Model already exists: {save_path}")

    X_train, X_val, y_train, y_val = load_data(args.dataset)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=(device.type == "cuda"),
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=(device.type == "cuda")
    )

    model = PricingNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print("\nTraining started...\n")
    start_time = time.time()

    for epoch in range(EPOCHS):

        # Training
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_dataset)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("Training produced an invalid loss.")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # The last epoch isn't necessarily the best one.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    training_time = time.time() - start_time

    # The learning curves might be useful for my thesis later.
    np.savez(
        MODEL_PATH / f"training_history_{args.dataset}.npz",
        train_loss=np.array(train_losses),
        val_loss=np.array(val_losses)
    )

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()