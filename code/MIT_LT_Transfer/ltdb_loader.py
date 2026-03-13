import numpy as np
import wfdb
import os
import torch
import glob
from scipy.signal import resample
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# 1. Configuration & Path Setup
# ==========================================
CONFIG = {
    # Ensure this matches your local folder structure shown in your screenshot
    "raw_dir": 'raw_data/ltdb/',      
    "save_dir": 'used_data/data_transfer_ltdb',
    "target_fs": 100,                 # Match PTB-XL sampling rate
    "window_seconds": 2.5,             # Match PTB-XL 10s window
    "stride_seconds": 1.75,              # 5s stride for 50% overlap
}

# Automatically detect all .hea files in the raw_dir
raw_files = glob.glob(os.path.join(CONFIG["raw_dir"], "*.hea"))
CONFIG["ltdb_records"] = [os.path.basename(f).replace('.hea', '') for f in raw_files]

print(f"Searching for records in: {CONFIG['raw_dir']}")
print(f"Detected records: {CONFIG['ltdb_records']}")

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. LTDB Specific Processing
# ==========================================

def process_ltdb_signals(record_name):
    path = os.path.join(CONFIG["raw_dir"], record_name)
    
    # 1. Load signal and annotations
    signal, fields = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, 'atr')
    
    original_fs = fields['fs']
    
    # 2. Resample Signal to 100Hz for consistency with PTB-XL
    num_samples_new = int(len(signal) * CONFIG["target_fs"] / original_fs)
    resampled_signal = resample(signal, num_samples_new)
    
    # 3. Sliding Window Parameters
    '''window_size = CONFIG["window_seconds"] * CONFIG["target_fs"]
    stride = CONFIG["stride_seconds"] * CONFIG["target_fs"]
    '''
    # 2. FORCE INTEGERS HERE
    # Use // for integer division or wrap in int()
    window_size = int(CONFIG["window_seconds"] * CONFIG["target_fs"]) # Should be 500
    stride = int(CONFIG["stride_seconds"] * CONFIG["target_fs"])      # Should be 250

    X_list = []
    y_list = []
    
    for start in range(0, len(resampled_signal) - window_size, stride):
        end = start + window_size
        
        # Map window back to original sample indices to check labels
        orig_start = int(start * original_fs / CONFIG["target_fs"])
        orig_end = int(end * original_fs / CONFIG["target_fs"])
        
        # Get annotations within this time window
        indices = np.where((ann.sample >= orig_start) & (ann.sample < orig_end))[0]
        window_anns = [ann.symbol[i] for i in indices]
        
        # STRICT PURITY CHECK: 
        # 1. Must have at least one annotation
        # 2. All annotations in the window must be identical (len(set) == 1)
        # 3. Force a fixed length and lead count.
        # 4. allow a window to be labeled as an arrhythmia if it is the most frequent label, even if it isn't 100% pure.
        if len(window_anns) > 0:
            counts = Counter(window_anns)
            label, count = counts.most_common(1)[0]
            
            # Logic: If it's Normal, we still want it pure.
            # If it's an Arrhythmia (any label != 'N'), 70% frequency is enough.
            is_pure = len(set(window_anns)) == 1
            is_majority_arrhythmia = (label != 'N') and (count / len(window_anns) >= 0.2)
            
            if is_pure or is_majority_arrhythmia:
                if label not in ['~', '|', '+', 'x']:
                    segment = resampled_signal[start:end].T
                    if segment.shape == (2, 250):
                        X_list.append(segment)
                        y_list.append([label])

    return X_list, y_list

# ==========================================
# 3. Execution & Data Splitting
# ==========================================

all_X = []
all_y = []

print("\nStarting LTDB Processing...")
for record in CONFIG["ltdb_records"]:
    try:
        X_parts, y_parts = process_ltdb_signals(record)
        all_X.extend(X_parts)
        all_y.extend(y_parts)
        print(f"Processed {record}: Found {len(X_parts)} strict-pure segments.")
    except Exception as e:
        print(f"Error processing {record}: {e}")

if len(all_X) == 0:
    print("\n!!! ERROR: No segments found. Check your 'Pure Mood' logic or data paths.")
else:
    # Convert to Numpy
    X_final = np.array(all_X)
    
    # Multi-label Binarizer (even for single labels, this keeps format consistent)
    mlb = MultiLabelBinarizer()
    y_final = mlb.fit_transform(all_y)

    # 80/20 Train-Test Split
    split_idx = int(len(X_final) * 0.8)
    indices = np.random.permutation(len(X_final))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    # ==========================================
    # 4. Saving Processed Data
    # ==========================================
    # We save as .pt files to be loaded directly by the transfer_train.py script
    torch.save({
        'X': torch.tensor(X_final[train_idx], dtype=torch.float32),
        'y': torch.tensor(y_final[train_idx], dtype=torch.float32),
        'classes': mlb.classes_
    }, os.path.join(CONFIG["save_dir"], 'ltdb_train.pt'))

    torch.save({
        'X': torch.tensor(X_final[test_idx], dtype=torch.float32),
        'y': torch.tensor(y_final[test_idx], dtype=torch.float32),
        'classes': mlb.classes_
    }, os.path.join(CONFIG["save_dir"], 'ltdb_test.pt'))

    print(f"\nLTDB Transfer Data Saved successfully to {CONFIG['save_dir']}!")
    print(f"Total Segments Found: {len(X_final)}")
    print(f"Classes (Arrhythmia types): {mlb.classes_}")
    
    # Print distribution so you can see if one class dominates
    label_counts = Counter([item for sublist in all_y for item in sublist])
    print(f"Class Distribution: {dict(label_counts)}")