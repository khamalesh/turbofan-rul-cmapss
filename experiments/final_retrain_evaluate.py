import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, random_split

from src.data_loader import load_cmapp_data
from src.preprocess_combined import preprocess_df
from src.dataset import RULDataset
from src.model import CNN_BiLSTM_Attention
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_residuals,
    save_metrics_to_excel,
)

# Best Optuna-tuned hyperparameters for each dataset (GRU removed)
best_hyperparams = {
    "train_FD001.txt": {
        "seq_length": 35, "batch_size": 16, "hidden_size": 256,
        "dropout": 0.1442, "lr": 0.00090117,
        "attn_type": "feature", "loss_fn": "composite"
    },
    "train_FD002.txt": {
        "seq_length": 35, "batch_size": 8, "hidden_size": 256,
        "dropout": 0.3092, "lr": 0.0004953,
        "attn_type": "feature", "loss_fn": "mse"
    },
    "train_FD003.txt": {
        "seq_length": 30, "batch_size": 16, "hidden_size": 128,
        "dropout": 0.1382, "lr": 1.929e-4,
        "attn_type": "feature", "loss_fn": "composite"
    },
    "train_FD004.txt": {
        "seq_length": 25, "batch_size": 8, "hidden_size": 128,
        "dropout": 0.1071, "lr": 5.098e-4,
        "attn_type": "dual", "loss_fn": "mse"
    }
}

def train_on_single_dataset(dataset_path, drop_columns, device, config):
    dataset_name = os.path.basename(dataset_path).replace("train_", "").replace(".txt", "")
    print(f"\n🔧 Training on: {dataset_name}")

    df = load_cmapp_data(dataset_path, drop_columns)
    inject_mode = (dataset_name == "FD004")
    df_processed = preprocess_df(df, dataset_name=dataset_name, inject_mode=inject_mode)

    feature_cols = [c for c in df_processed.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
    aux_cols = ['mode_id'] if inject_mode else None

    dataset = RULDataset(df_processed, seq_length=config["seq_length"],
                         feature_cols=feature_cols, aux_cols=aux_cols)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CNN_BiLSTM_Attention(
        input_dim=len(feature_cols),
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        attn_type=config["attn_type"]
    ).to(device)

    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.get("epochs", 50),
        lr=config["lr"],
        loss_fn=config.get("loss_fn", "mse"),
        patience=config.get("patience", 5),
        dataset_name=dataset_name,
        save_best_path=f"checkpoints/best_model_{dataset_name}.pth"
    )

    preds, targets, metrics = evaluate_model(model, val_loader, device)
    save_metrics_to_excel(dataset_name, metrics['rmse'], metrics['mae'], metrics['r2'])

    print(f"📊 Validation RMSE: {metrics['rmse']:.4f}")

    os.makedirs("plots", exist_ok=True)
    plot_actual_vs_predicted(targets, preds, save_path=f"plots/actual_vs_predicted_{dataset_name}.png")
    plot_residuals(targets, preds, save_path=f"plots/residuals_{dataset_name}.png")
    print(f"📁 Saved plots to plots/")

    return model, metrics['rmse'], metrics['mae'], metrics['r2'], len(train_losses)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NASA TURBOFAN ENGINE RUL PREDICTION")
    print("Hybrid CNN-BiLSTM-Attention Model | RMSE Optimization")
    print("Author : Khamalesh Ramesh")
    print("College: National College of Ireland")
    print("ID     : x23325216")
    print("=" * 60)
    

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"🖥️ Using device: {device}")

    drop_columns = [
        'operational_setting_1', 'operational_setting_2',
        'sensor_measurement_1', 'sensor_measurement_5', 'sensor_measurement_6',
        'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_14',
        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
    ]

    dataset_files = [
        "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD001.txt",
        "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD002.txt",
        "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD003.txt",
        "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD004.txt"
    ]

    final_summary = []

    for path in dataset_files:
        filename = os.path.basename(path)
        if filename in best_hyperparams:
            config = best_hyperparams[filename]
            config.update({
                "loss_fn": "composite" if filename == "train_FD004.txt" else "mse",
                "attn_type": "dual",
                "epochs": 50,
                "patience": 5
            })

            model, rmse, mae, r2, trained_epochs = train_on_single_dataset(
                dataset_path=path,
                drop_columns=drop_columns,
                device=device,
                config=config
            )

            final_summary.append({
                "Dataset": filename.replace("train_", "").replace(".txt", ""),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                "Epochs Trained": trained_epochs,
                "Loss Function": config["loss_fn"],
                "GRU": "No",
                "Attention": config["attn_type"]
            })
        else:
            print(f"⚠️ Skipping {filename} — no hyperparameters found.")

    os.makedirs("plots", exist_ok=True)
    summary_df = pd.DataFrame(final_summary)
    summary_df.to_csv("plots/final_model_summary.csv", index=False)

    print("\n📊 Composite Evaluation Summary:")
    print(summary_df.to_string(index=False))
