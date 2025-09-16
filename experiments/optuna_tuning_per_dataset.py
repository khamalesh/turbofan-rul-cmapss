import optuna
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split

from src.data_loader import load_cmapp_data
from src.preprocess_combined import preprocess_df
from src.dataset import RULDataset
from src.evaluate import evaluate_model
from experiments.final_retrain_evaluate import train_on_single_dataset

# 🚀 Device configuration
device = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available()
                     else "cpu")

# 🔧 Columns to drop (based on domain knowledge)
drop_columns = [
    'operational_setting_1', 'operational_setting_2',
    'sensor_measurement_1', 'sensor_measurement_5', 'sensor_measurement_6',
    'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_14',
    'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
]

# 📁 Dataset file paths (only FD003 and FD004)
dataset_files = [
    "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD001.txt"
    "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD002.txt"
    "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD003.txt",
    "/Users/khamaleshramesh/Documents/Research Topic/NASA/DATASET/CMAPSSData/train_FD004.txt"
]

# 🎯 Objective function factory
def create_objective(dataset_path):
    def objective(trial):
        config = {
            "seq_length": trial.suggest_categorical("seq_length", [25, 30, 35]),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.25),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "use_gru": trial.suggest_categorical("use_gru", [True, False]),
            "attn_type": trial.suggest_categorical("attn_type", ["feature", "dual"]),
            "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "composite"]),
            "epochs": 20,
            "patience": 3,
            "verbose": False
        }

        try:
            dataset_name = os.path.basename(dataset_path).replace("train_", "").replace(".txt", "")
            inject_mode = (dataset_name == "FD004")

            df = load_cmapp_data(dataset_path, drop_columns)
            df_processed = preprocess_df(df, dataset_name=dataset_name, inject_mode=inject_mode)

            feature_cols = [c for c in df_processed.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
            aux_cols = ['mode_id'] if inject_mode else None

            dataset = RULDataset(df_processed, seq_length=config["seq_length"],
                                 feature_cols=feature_cols, aux_cols=aux_cols)
            val_size = int(0.2 * len(dataset))
            _, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

            model, _, _, _ = train_on_single_dataset(
                dataset_path=dataset_path,
                drop_columns=drop_columns,
                device=device,
                config=config
            )

            preds, targets, metrics = evaluate_model(model, val_loader, device)
            rmse = metrics["rmse"]

            print(f"✅ Trial {trial.number:02d} | {dataset_name} | RMSE: {rmse:.4f}")
            return rmse

        except Exception as e:
            with open("optuna_failure_log.txt", "a") as f:
                f.write(f"Trial {trial.number} | {dataset_path} | ERROR: {str(e)}\n")
            return float("inf")

    return objective

# 🚦 Launch Optuna
if __name__ == "__main__":
    for dataset_path in dataset_files:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n🚀 Starting Optuna tuning for: {dataset_name}")

        study = optuna.create_study(direction="minimize")
        objective = create_objective(dataset_path)
        study.optimize(objective, n_trials=15, timeout=1800)

        best_trial = study.best_trial
        print(f"\n🏆 Best trial for {dataset_name}:")
        print(f"   RMSE: {best_trial.value:.4f}")
        print(f"   Parameters:")
        for key, value in best_trial.params.items():
            print(f"     {key}: {value}")
