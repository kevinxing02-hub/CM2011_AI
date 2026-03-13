import pandas as pd
import numpy as np
import wfdb
import ast
import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# 1. Preprocessing Interfaces (Placeholders)
# ==========================================

def denoise_signal(x):
    """
    Interface for Task 2: Denoising logic (e.g., bandpass filter).
    Currently returns signal as-is.
    """
    # TODO: Implement filtering here
    return x

def resample_signal(x, original_fs, target_fs):
    """
    Interface for Task 2: Resampling logic.
    Currently returns signal as-is.
    """
    if original_fs == target_fs:
        return x
    # TODO: Implement resampling here (e.g., using scipy.signal.resample)
    return x

# ==========================================
# 2. Raw Data Loading
# ==========================================

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    
    # Extract signal and metadata
    signals = np.array([signal for signal, meta in data])
    return signals

# Path and sampling rate setup
path = 'raw_data/ptbxl/'
sampling_rate = 100
save_dir = 'used_data/data_basic_ptb'
os.makedirs(save_dir, exist_ok=True)

# Load database and convert annotation strings to dicts
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load diagnostic aggregation rules
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# ==========================================
# 3. Label Binarization & Splitting
# ==========================================

mlb = MultiLabelBinarizer()
# We binarize the full set first to ensure consistent label indices
Y_bin = mlb.fit_transform(Y.diagnostic_superclass)

test_fold = 10
train_idx = np.where(Y.strat_fold != test_fold)[0]
test_idx = np.where(Y.strat_fold == test_fold)[0]

X_train, y_train = X[train_idx], Y_bin[train_idx]
X_test, y_test = X[test_idx], Y_bin[test_idx]

# ==========================================
# 4. Dataset Class & Saving Logic
# ==========================================

class ECGDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None):
        # Store as Tensor: [Batch, Leads, Time]
        # Transpose from (N, Time, Leads) to (N, Leads, Time)
        self.X = torch.tensor(X_data, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        # Apply preprocessing interfaces
        x = denoise_signal(x)
        # Note: resample usually happens before tensor conversion, 
        # but the interface is here if needed.
        
        if self.transform:
            x = self.transform(x)
        return x, y

# Instantiate
train_ds = ECGDataset(X_train, y_train)
test_ds = ECGDataset(X_test, y_test)

# Save Processed Data for later use (e.g., when moving to LTDB)
torch.save({
    'X': train_ds.X, 
    'y': train_ds.y, 
    'classes': mlb.classes_
}, os.path.join(save_dir, 'ptbxl_train.pt'))

torch.save({
    'X': test_ds.X, 
    'y': test_ds.y, 
    'classes': mlb.classes_
}, os.path.join(save_dir, 'ptbxl_test.pt'))

print(f"Data saved successfully. Classes found: {mlb.classes_}")

# ==========================================
# 5. DataLoaders
# ==========================================

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Ready for Model implementation