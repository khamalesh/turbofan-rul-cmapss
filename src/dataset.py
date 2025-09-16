import torch
from torch.utils.data import Dataset

class RULDataset(Dataset):
    def __init__(self, df, seq_length, feature_cols, aux_cols=None):
        """
        Dataset constructor for NASA CMAPSS Remaining Useful Life prediction.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame with sensor and label data.
            seq_length (int): Length of time-series input sequences.
            feature_cols (list): Columns to use as time-series features.
            aux_cols (list or None): Optional list of auxiliary column names 
                                     (e.g., ['mode_id']) to include at output time.
        """
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.aux_cols = aux_cols if aux_cols is not None else []
        self.data = df
        self.units = df['unit_number'].unique()
        self.sequences = []
        self.labels = []
        self.auxiliary = []

        self._prepare_sequences()

    def _prepare_sequences(self):
        for unit in self.units:
            unit_df = self.data[self.data['unit_number'] == unit].reset_index(drop=True)
            total_steps = len(unit_df)

            if total_steps < self.seq_length:
                continue  # Skip units that are too short for one full sequence

            for start_idx in range(total_steps - self.seq_length + 1):
                end_idx = start_idx + self.seq_length
                seq = unit_df.loc[start_idx:end_idx - 1, self.feature_cols].values
                label = unit_df.loc[end_idx - 1, 'RUL']

                self.sequences.append(seq)
                self.labels.append(label)

                if self.aux_cols:
                    aux = unit_df.loc[end_idx - 1, self.aux_cols].values
                    self.auxiliary.append(aux)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

        if self.aux_cols:
            sample['aux'] = torch.tensor(self.auxiliary[idx], dtype=torch.float32)

        return sample
