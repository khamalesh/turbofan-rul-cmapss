import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

def load_cmapp_data(filepath, drop_cols):
    cols = ['unit_number', 'time_in_cycles'] + \
           [f'operational_setting_{i}' for i in range(1, 4)] + \
           [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.drop(columns=[26, 27], inplace=True)  # Drop extra unnamed columns
    df.columns = cols
    df = df.drop(columns=drop_cols)
    return df

def wavelet_denoise(signal, wavelet='db4', level=1, eps=1e-8):
    signal = np.array(signal)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level]))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    new_coeff = []
    for c in coeff[1:]:
        magnitude = np.abs(c)
        magnitude_safe = np.where(magnitude < eps, eps, magnitude)
        thresholded = (1 - uthresh / magnitude_safe)
        thresholded = np.maximum(thresholded, 0) * c
        new_coeff.append(thresholded)

    coeff[1:] = new_coeff
    denoised = pywt.waverec(coeff, wavelet, mode='symmetric')
    return denoised[:len(signal)]

def apply_wavelet_denoising(df, features):
    df_copy = df.copy()
    for unit in df_copy['unit_number'].unique():
        mask = df_copy['unit_number'] == unit
        for feat in features:
            signal = df_copy.loc[mask, feat].values
            if len(signal) > 10:
                denoised_signal = wavelet_denoise(signal)
                df_copy.loc[mask, feat] = denoised_signal
    return df_copy

def piecewise_rul_labeling(df, threshold=125):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL_raw'] = max_cycle - df['time_in_cycles']
    df['RUL'] = df['RUL_raw'].clip(upper=threshold)
    df.drop(columns=['RUL_raw'], inplace=True)
    return df

def feature_engineering(df, features, rolling_window=10):
    df_copy = df.copy()
    for feat in features:
        df_copy[f'{feat}_roll_mean'] = df_copy.groupby('unit_number')[feat].transform(
            lambda x: x.rolling(rolling_window, min_periods=1).mean())
        df_copy[f'{feat}_roll_std'] = df_copy.groupby('unit_number')[feat].transform(
            lambda x: x.rolling(rolling_window, min_periods=1).std().fillna(0))
        df_copy[f'{feat}_delta'] = df_copy.groupby('unit_number')[feat].diff().fillna(0)
    return df_copy

def inject_operational_mode(df):
    # Categorize op_mode using operational_setting_3
    mode_map = {v: i for i, v in enumerate(sorted(df['operational_setting_3'].unique()))}
    df['mode_id'] = df['operational_setting_3'].map(mode_map).astype(int)
    return df

def preprocess_pipeline(filepath, inject_mode=False):
    drop_columns = [
        'operational_setting_1', 'operational_setting_2',
        'sensor_measurement_1', 'sensor_measurement_5', 'sensor_measurement_6',
        'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_14',
        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
    ]

    # 1. Load data
    df = load_cmapp_data(filepath, drop_columns)

    # 2. Inject operational mode ID if needed
    if inject_mode:
        df = inject_operational_mode(df)

    # 3. Normalize features
    feature_cols = df.columns.drop(['unit_number', 'time_in_cycles'])
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    pt = PowerTransformer(method='yeo-johnson')
    df[feature_cols] = pt.fit_transform(df[feature_cols])

    # 4. Wavelet denoising
    sensor_features = df.columns.drop(['unit_number', 'time_in_cycles'])
    df = apply_wavelet_denoising(df, sensor_features)

    # 5. Label RUL
    df = piecewise_rul_labeling(df)

    # 6. Feature engineering
    features_for_fe = df.columns.drop(['unit_number', 'time_in_cycles', 'RUL'])
    df = feature_engineering(df, features_for_fe)

    return df
