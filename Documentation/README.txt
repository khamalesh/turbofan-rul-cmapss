# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

This project presents a research-grade pipeline for Remaining Useful Life (RUL) prediction using the NASA CMAPSS turbofan engine dataset. The approach integrates a hybrid deep learning architecture with domain-aware preprocessing, composite loss formulation, and automated hyperparameter optimization.

---

## Project Structure

```
rul-prediction/
├── data/                          # Raw CMAPSS datasets and documentation
│   ├── train_FD001.txt            # Training data for FD001
│   ├── test_FD001.txt             # Test data for FD001
│   ├── RUL_FD001.txt              # Ground truth RUL for FD001
│   └── ...                        # Other FD002–FD004 subsets
│
├── src/                           # Core modules and helper functions
│   ├── model.py                   # CNN-BiLSTM-Attention model definition
│   ├── data_loader.py             # Data loading and normalization logic
│   ├── dataset.py                 # PyTorch Dataset class for RUL
│   ├── train.py                   # Training loop and logging
│   ├── evaluate.py                # Evaluation metrics and visualization
│   ├── losses.py                  # Composite loss implementation
│   └── preprocess_combined.py     # Full preprocessing pipeline
│
├── experiments/                   # Experiment scripts and outputs
│   ├── final_retrain_evaluate.py  # Final model retraining and evaluation
│   ├── optuna_tuning_per_dataset.py # Optuna hyperparameter tuning
│   ├── checkpoints/               # Best model weights (.pth files)
│   ├── loss_logs/                 # Training/validation loss logs (.csv)
│   ├── plots/                     # Model evaluation visualizations (.png)
│   ├── results/                   # Excel summaries of metrics
│   └── snapshots/                 # Intermediate data outputs per step
│
├── requirements.txt               # Python package dependencies
└── README.md                      # Project overview and usage guide
```

---

## Model Architecture

- CNN Feature Extraction: Multi-scale convolution layers (kernel sizes 3, 5, 7)
- BiLSTM Backbone: Captures temporal engine degradation patterns
- Dual Attention Mechanism:
  - Feature-level attention
  - Temporal-level attention
- Regression Head: Two-layer fully connected prediction module
- Loss Function: Composite loss (weighted MSE and Huber loss)

---

## Dataset

The project utilizes all four CMAPSS subsets:

- FD001: One operating condition, one fault mode  
- FD002: Multiple operating conditions, one fault mode  
- FD003: One operating condition, multiple fault modes  
- FD004: Multiple operating conditions, multiple fault modes  

All dataset files are placed in the `./data/` directory and include:
- `train_FDxxx.txt`
- `test_FDxxx.txt`
- `RUL_FDxxx.txt`

---

## Execution Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Final Models

```bash
python -m experiments.final_retrain_evaluate
```

### 3. Hyperparameter Tuning (Optuna)

```bash
python -m experiments.optuna_tuning_per_dataset
```

---

## Output and Results

After execution, the following outputs are generated:

- checkpoints/: Best model weights per dataset (.pth)
- loss_logs/: Per-epoch training and validation losses (CSV format)
- plots/: Evaluation plots (actual vs predicted, residuals)
- results/: Metric summaries (RMSE, MAE, R²) in Excel format
- snapshots/: Processed intermediate dataframes (scaling, transformation, denoising, RUL labeling, etc.)

---

## Dependencies

See `requirements.txt` for all dependencies. Major packages include:

- torch
- optuna
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Author

Khamalesh Ramesh  
MSc in Data Analytics  

