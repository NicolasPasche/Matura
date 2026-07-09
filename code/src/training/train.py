import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
import time
from pathlib import Path


# Project paths

project_root = Path(__file__).parent.parent.parent.parent

sys.path.append(str(project_root / "code" / "src"))

from models.network import PricingNetwork


data_path = project_root / "data" / "processed"

model_path = project_root / "models" / "best_model_dividend.pth"

model_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# Load processed data

X_train = np.load(
    data_path / "X_train2.npy"
)

X_val = np.load(
    data_path / "X_val2.npy"
)

y_train = np.load(
    data_path / "y_train2.npy"
)

y_val = np.load(
    data_path / "y_val2.npy"
)


print("\nTraining data:")
print(X_train.shape)
print(y_train.shape)


# Convert data to tensors

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)


print("\nTensor shape:")
print(X_train.shape)


# Create datasets

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


# Data loaders

BATCH_SIZE = 4096


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)


# Initialize model

model = PricingNetwork().to(device)


print("\nModel:")
print(model)

print(
    f"\nNumber of parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# Loss and optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training parameters

EPOCHS = 100

best_val_loss = float("inf")

train_losses = []
val_losses = []


print("\nTraining started...\n")


start_time = time.time()


# Training loop

for epoch in range(EPOCHS):

    model.train()

    running_train_loss = 0.0


    for X_batch, y_batch in train_loader:


        X_batch = X_batch.to(
            device,
            non_blocking=True
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True
        )


        predictions = model(X_batch)


        loss = criterion(
            predictions,
            y_batch
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_train_loss += loss.item()


    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # Validation

    model.eval()

    running_val_loss = 0.0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:


            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )


            predictions = model(X_batch)


            loss = criterion(
                predictions,
                y_batch
            )


            running_val_loss += loss.item()


    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


    # Save best model

    if val_loss < best_val_loss:


        best_val_loss = val_loss


        torch.save(
            model.state_dict(),
            model_path
        )


        print("Saved best model.")



training_time = time.time() - start_time


print("\nTraining finished.")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Model saved to: {model_path}")