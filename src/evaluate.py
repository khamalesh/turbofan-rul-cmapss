import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on the given data_loader.

    Returns:
        preds (np.ndarray)
        targets (np.ndarray)
        metrics (dict): {'rmse': ..., 'mae': ..., 'r2': ...}
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            xb = batch['sequence'].to(device)
            yb = batch['label'].to(device)
            aux_input = batch.get('aux', None)
            if aux_input is not None:
                aux_input = aux_input.to(device)

            outputs = model(xb, aux_input) if aux_input is not None else model(xb)
            preds.extend(outputs.cpu().numpy())
            targets.extend(yb.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"\n🛰️ Evaluation Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    return preds, targets, metrics

def save_metrics_to_excel(dataset_name, rmse, mae, r2, path="results/rul_results_summary.xlsx"):
    """
    Saves the evaluation metrics to an Excel file with a separate sheet per dataset.
    """
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame([{
        "Dataset": dataset_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }])

    if os.path.exists(path):
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=dataset_name, index=False)
    else:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=dataset_name, index=False)

def plot_actual_vs_predicted(targets, preds, save_path=None):
    """
    Plots predicted vs. actual RUL values.

    Args:
        targets (np.ndarray): Ground truth RUL values
        preds (np.ndarray): Predicted RUL values
        save_path (str or None): If provided, saves the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(targets, preds, alpha=0.5)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Actual vs Predicted RUL")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig

def plot_residuals(targets, preds, save_path=None):
    """
    Plots residuals (Actual - Predicted).

    Args:
        targets (np.ndarray): Ground truth RUL values
        preds (np.ndarray): Predicted RUL values
        save_path (str or None): If provided, saves the figure
    """
    residuals = targets - preds
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(preds, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted RUL")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig
