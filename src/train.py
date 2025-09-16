import os
import torch
import numpy as np
import pandas as pd
from src.losses import CompositeLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast
from torch.cuda.amp import GradScaler

def save_loss_log_to_csv(train_losses, val_losses, dataset_name="FDXXX", log_dir="loss_logs"):
    """
    Saves training and validation loss histories to CSV.
    """
    os.makedirs(log_dir, exist_ok=True)
    file_path = f"{log_dir}/{dataset_name}_losses.csv"
    df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    df.to_csv(file_path, index=False)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-3,
    loss_fn='mse',
    patience=5,
    verbose=True,
    dataset_name="FDXXX",
    save_best_path=None,
    use_amp=False
):
    """
    Trains the given model and returns the trained model and loss history.

    Parameters:
        model: PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: CUDA/MPS/CPU device
        epochs: Training epochs
        lr: Learning rate
        loss_fn: 'mse' or 'composite'
        patience: Early stopping patience
        verbose: Whether to print training progress
        dataset_name: For naming log files
        save_best_path: Optional checkpoint path
        use_amp: Use mixed precision
    """
    # Select loss function
    if loss_fn == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'composite':
        criterion = CompositeLoss(alpha=0.5, delta=1.0)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler() if use_amp else None

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Create checkpoint folder if needed
    if save_best_path:
        os.makedirs(os.path.dirname(save_best_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            xb = batch['sequence'].to(device)
            yb = batch['label'].to(device)
            aux_input = batch.get('aux', None)
            if aux_input is not None:
                aux_input = aux_input.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type, enabled=True):
                    preds = model(xb, aux_input) if aux_input is not None else model(xb)
                    loss = criterion(preds, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(xb, aux_input) if aux_input is not None else model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch['sequence'].to(device)
                    yb = batch['label'].to(device)
                    aux_input = batch.get('aux', None)
                    if aux_input is not None:
                        aux_input = aux_input.to(device)

                    with autocast(device_type=device.type, enabled=use_amp):
                        preds = model(xb, aux_input) if aux_input is not None else model(xb)
                        loss = criterion(preds, yb)

                    val_running_loss += loss.item() * xb.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if verbose:
                print(f"Epoch {epoch:03d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping & checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best_path:
                    torch.save(model.state_dict(), save_best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch:03d} - Train Loss: {train_loss:.4f}")

    save_loss_log_to_csv(train_losses, val_losses, dataset_name)
    return model, train_losses, val_losses
