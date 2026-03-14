"""
PIPELINE STEP 1: PTB-XL Data Loading & Preprocessing
Description:
    This script loads the raw PTB-XL dataset, aggregates diagnostic labels into 
    superclasses, and saves the signals as PyTorch (.pt) tensors.
    
Requirements:
    - Raw data should be in: 'raw_data/ptbxl/'
    - Files needed: 'ptbxl_database.csv', 'scp_statements.csv', and the signal folders.
"""

import pandas as pd
import numpy as np
import wfdb
import ast
import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# 1. Configuration & Relative Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'raw_data', 'ptbxl', '')
SAVE_DIR = os.path.join(BASE_DIR, 'used_data', 'data_basic_ptb')
SAMPLING_RATE = 100

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Preprocessing Interfaces
# ==========================================

def denoise_signal(x):
    """Placeholder for Task 2: Implement bandpass filtering here."""
    return x

def resample_signal(x, original_fs, target_fs):
    """Placeholder for Task 2: Implement resampling logic here."""
    if original_fs == target_fs:
        return x
    return x

# ==========================================
# 3. Data Loading Logic
# ==========================================

def load_raw_data(df, sampling_rate, path):
    """Reads wfdb records based on the filename columns in the dataframe."""
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    
    signals = np.array([signal for signal, meta in data])
    return signals

def aggregate_diagnostic(y_dic, agg_df):
    """Maps specific SCP codes to broader diagnostic superclasses."""
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# ==========================================
# 4. Dataset Class
# ==========================================

class ECGDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None):
        # Transpose from (N, Time, Leads) to (N, Leads, Time) for CNN input
        self.X = torch.tensor(X_data, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        x = denoise_signal(x)
        if self.transform:
            x = self.transform(x)
        return x, y

# ==========================================
# 5. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
    print(f"Checking for data in: {RAW_DATA_PATH}")
    
    # Load metadata
    Y = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, SAMPLING_RATE, RAW_DATA_PATH)

    # Load diagnostic aggregation rules
    agg_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    # Label Binarization
    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(Y.diagnostic_superclass)

    # Train/Test Split (Fold 10 is typically the recommended test set for PTB-XL)
    test_fold = 10
    train_idx = np.where(Y.strat_fold != test_fold)[0]
    test_idx = np.where(Y.strat_fold == test_fold)[0]

    X_train, y_train = X[train_idx], Y_bin[train_idx]
    X_test, y_test = X[test_idx], Y_bin[test_idx]

    # Create and Save Datasets
    train_ds = ECGDataset(X_train, y_train)
    test_ds = ECGDataset(X_test, y_test)

    torch.save({'X': train_ds.X, 'y': train_ds.y, 'classes': mlb.classes_}, 
               os.path.join(SAVE_DIR, 'ptbxl_train.pt'))

    torch.save({'X': test_ds.X, 'y': test_ds.y, 'classes': mlb.classes_}, 
               os.path.join(SAVE_DIR, 'ptbxl_test.pt'))

    print(f"Successfully saved processed data to {SAVE_DIR}")
    print(f"Classes found: {mlb.classes_}")