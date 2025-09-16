import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from src.data_loader import (
    apply_wavelet_denoising,
    piecewise_rul_labeling,
    feature_engineering,
    inject_operational_mode  # New: Optional mode encoding hook
)

def save_df_snapshot(df: pd.DataFrame, step_name: str, dataset_name: str, snapshot_dir: str = "snapshots/df_snapshots"):
    """
    Save the current state of the DataFrame to a CSV for debug or audit trails.

    Parameters:
        df (pd.DataFrame): Data to save.
        step_name (str): The step identifier.
        dataset_name (str): Dataset code (e.g., FD001).
        snapshot_dir (str): Where to store snapshots.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = f"{snapshot_dir}/{dataset_name}_{step_name}.csv"
    df.to_csv(filename, index=False)

def preprocess_df(df: pd.DataFrame, dataset_name: str = "unknown", inject_mode: bool = False) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
        - Optional operational mode injection
        - MinMax scaling
        - Power transform
        - Wavelet denoising
        - RUL labeling
        - Temporal feature engineering

    Parameters:
        df (pd.DataFrame): Raw input data.
        dataset_name (str): Name for snapshot files.
        inject_mode (bool): Whether to inject 'mode_id' from operational settings.

    Returns:
        pd.DataFrame: Fully processed DataFrame.
    """

    # STEP 0: Optional Operational Mode Injection
    if inject_mode:
        df = inject_operational_mode(df)
        save_df_snapshot(df, "step0_mode_injected", dataset_name)

    # STEP 1: MinMax Scaling
    feature_cols = df.columns.drop(['unit_number', 'time_in_cycles'])
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    save_df_snapshot(df, "step1_scaled", dataset_name)

    # STEP 2: PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    df[feature_cols] = pt.fit_transform(df[feature_cols])
    save_df_snapshot(df, "step2_power_transformed", dataset_name)

    # STEP 3: Wavelet Denoising
    sensor_features = df.columns.drop(['unit_number', 'time_in_cycles'])
    df = apply_wavelet_denoising(df, sensor_features)
    save_df_snapshot(df, "step3_wavelet", dataset_name)

    # STEP 4: RUL Labeling
    df = piecewise_rul_labeling(df)
    save_df_snapshot(df, "step4_rul", dataset_name)

    # STEP 5: Feature Engineering (Rolling stats, deltas, etc.)
    features_for_fe = df.columns.drop(['unit_number', 'time_in_cycles', 'RUL'])
    df = feature_engineering(df, features_for_fe)
    save_df_snapshot(df, "step5_features", dataset_name)

    return df
