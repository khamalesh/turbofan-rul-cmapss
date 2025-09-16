Turbofan RUL Prediction (NASA C-MAPSS)

Overview
This repository contains a deep learning framework for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.
A hybrid CNN–BiLSTM–Attention model is implemented to capture both local degradation patterns and long-term temporal dependencies.
The objective of this project is to demonstrate the application of advanced machine learning techniques to predictive maintenance in aerospace systems.

Technology Stack
Language: Python 3.10
Framework: PyTorch (Apple MPS backend)
Libraries: NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Optuna, PyWavelets
Tools: JupyterLab, Visual Studio Code

Repository Structure:
The project is organized as follows:
├── checkpoints # Saved model weights for each dataset
│ ├── best_model_FD001.pth
│ ├── best_model_FD002.pth
│ ├── best_model_FD003.pth
│ └── best_model_FD004.pth
├── data # CMAPSS dataset files & reference docs
│ ├── Damage Propagation Modeling.pdf
│ ├── readme.txt
│ ├── RUL_FD001.txt
│ ├── RUL_FD002.txt
│ ├── RUL_FD003.txt
│ ├── RUL_FD004.txt
│ ├── test_FD001.txt
│ ├── test_FD002.txt
│ ├── test_FD003.txt
│ ├── test_FD004.txt
│ ├── train_FD001.txt
│ ├── train_FD002.txt
│ ├── train_FD003.txt
│ └── train_FD004.txt
├── experiments # Training & tuning scripts
│ ├── final_retrain_evaluate.py
│ └── optuna_tuning_per_dataset.py
├── loss_logs # Per-epoch training and validation loss records
│ ├── FD001_losses.csv
│ ├── FD002_losses.csv
│ ├── FD003_losses.csv
│ └── FD004_losses.csv
├── plots # Model evaluation plots & summaries
│ ├── actual_vs_predicted_FD001.png
│ ├── actual_vs_predicted_FD002.png
│ ├── actual_vs_predicted_FD003.png
│ ├── actual_vs_predicted_FD004.png
│ ├── final_model_summary.csv
│ ├── residuals_FD001.png
│ ├── residuals_FD002.png
│ ├── residuals_FD003.png
│ └── residuals_FD004.png
├── README.txt # Project overview and basic instructions
├── requirements.txt # Python dependencies
├── results # Final prediction results & summaries
│ └── rul_results_summary.xlsx
├── snapshots # Preprocessed data snapshots
│ └── df_snapshots
└── src # Core source code modules
├── data_loader.py # Data loading, denoising, labeling, feature extraction
├── dataset.py # Sequence generation and dataset formatting
├── evaluate.py # Model evaluation and visualization
├── losses.py # Loss function definitions
├── model.py # CNN–BiLSTM–Attention architecture
├── preprocess_combined.py # Combined preprocessing steps
└── train.py # Training loop, early stopping, checkpointing


Key Features
* Wavelet-based denoising for noisy sensor signals
* Temporal feature engineering: rolling mean, rolling standard deviation, and delta differences
* Hybrid architecture combining CNN layers, BiLSTM layers, and dual attention mechanisms
* Composite loss function (MSE + Huber) for robust learning
* Automated hyperparameter tuning with Optuna across dataset subsets
* Reproducible outputs including plots, logs, and snapshots

How to Run
Clone the repository:
git clone https://github.com/<YOUR_USERNAME>/turbofan-rul-cmapss.git
cd turbofan-rul-cmapss


Install dependencies:
pip install -r requirements.txt


Download the dataset:
Source: NASA Prognostics Data Repository
Place the files inside the data/ directory (e.g., train_FD001.txt, test_FD001.txt, etc.).

Train and evaluate the model:
python experiments/final_retrain_evaluate.py


Run hyperparameter tuning:
python experiments/optuna_tuning_per_dataset.py


Results
Performance across C-MAPSS subsets using tuned configurations:

Dataset	RMSE	MAE	R²	Loss Function	Attention
FD001	0.7655	0.5354	0.9997	MSE	   Dual
FD002	1.7763	1.2934	0.9982	MSE	   Dual
FD003	1.3242	0.9195	0.9989	MSE	   Dual
FD004	3.0947	2.0984	0.9943	Composite	Dual


References
NASA C-MAPSS Turbofan Engine Degradation Data – Link (https://data.nasa.gov/dataset/C-MAPSS/)
Guo et al. (2023), Zhang et al. (2024) – Hybrid attention-based RUL models
Wu et al. (2020) – BiLSTM for C-MAPSS
Yildirim & Rana (2024) – Hyperparameter tuning with Optuna

Author
Khamalesh Ramesh
MSc Data Analytics, National College of Ireland
