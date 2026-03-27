"""
PIPELINE STEP 4: LTDB Sequential Data Loading with Targeted Augmentation (FIXED)
"""
import numpy as np
import wfdb
import os
import torch
import glob
from scipy.signal import resample

CONFIG = {
    "target_fs": 100,
    "window_seconds": 10,  # 10s Window
    "stride_seconds": 5,   # 5s Stride for overlap
    "max_beats": 20,       # Increased for 10s window
    "aug_offsets": [-0.2, 0.2],
}

LABEL_MAP = {
    'N': 0, 'L': 0, 'R': 0,
    'V': 1, 'E': 1,
    'A': 2, 'a': 2, 'J': 2, 'S': 2,
    'f': 3, 'F': 3, 'Q': 3,
}

RARE_CLASSES = [2, 3]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this matches your actual directory structure
RAW_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'raw_data', 'ltdb'))
SAVE_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb_10s')

def get_padded_label(window_labels):
    if len(window_labels) < CONFIG["max_beats"]:
        return window_labels + [-1] * (CONFIG["max_beats"] - len(window_labels))
    return window_labels[:CONFIG["max_beats"]]

def process_ltdb_signals(record_full_path):
    # wfdb.rdsamp/rdann take the path WITHOUT extension
    record_base = record_full_path.replace('.hea', '')
    
    signal, fields = wfdb.rdsamp(record_base)
    ann = wfdb.rdann(record_base, 'atr')
    original_fs = fields['fs']
    
    num_samples_new = int(len(signal) * CONFIG["target_fs"] / original_fs)
    resampled_signal = resample(signal, num_samples_new)
    
    win_size = int(CONFIG["window_seconds"] * CONFIG["target_fs"]) # 1000
    stride = int(CONFIG["stride_seconds"] * CONFIG["target_fs"])

    X_list, y_list = [], []
    processed_starts = set()

    # --- 1. Standard Sliding Window ---
    for start in range(0, len(resampled_signal) - win_size, stride):
        end = start + win_size
        orig_start, orig_end = int(start * original_fs / 100), int(end * original_fs / 100)
        
        idx = np.where((ann.sample >= orig_start) & (ann.sample < orig_end))[0]
        labels = [LABEL_MAP[ann.symbol[i]] for i in idx if ann.symbol[i] in LABEL_MAP]
        
        if labels:
            seg = resampled_signal[start:end].T # Shape: [2, 1000]
            if seg.shape == (2, 1000): # FIX: Updated for 10s
                X_list.append(seg.astype(np.float32))
                y_list.append(np.array(get_padded_label(labels), dtype=np.int64))
                processed_starts.add(start)

    # --- 2. Targeted Augmentation ---
    for i, symbol in enumerate(ann.symbol):
        label = LABEL_MAP.get(symbol)
        if label in RARE_CLASSES:
            for offset_sec in CONFIG["aug_offsets"]:
                center_samp_100 = int(ann.sample[i] * CONFIG["target_fs"] / original_fs)
                start = center_samp_100 + int(offset_sec * 100) - (win_size // 2)
                end = start + win_size
                
                if start < 0 or end > len(resampled_signal) or start in processed_starts:
                    continue
                
                orig_s, orig_e = int(start * original_fs / 100), int(end * original_fs / 100)
                idx = np.where((ann.sample >= orig_s) & (ann.sample < orig_e))[0]
                labels = [LABEL_MAP[ann.symbol[j]] for j in idx if ann.symbol[j] in LABEL_MAP]
                
                if labels:
                    seg = resampled_signal[start:end].T
                    if seg.shape == (2, 1000): # FIX: Updated for 10s
                        X_list.append(seg.astype(np.float32))
                        y_list.append(np.array(get_padded_label(labels), dtype=np.int64))
                        processed_starts.add(start)

    return X_list, y_list

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # FIX: Use recursive=True to find files in subfolders (01/010/ etc)
    search_pattern = os.path.join(RAW_DATA_PATH, "**", "*.hea")
    record_paths = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(record_paths)} records in {RAW_DATA_PATH}")

    all_X, all_y = [], []
    for path in record_paths:
        record_id = os.path.basename(path).replace('.hea', '')
        try:
            X_p, y_p = process_ltdb_signals(path)
            all_X.extend(X_p)
            all_y.extend(y_p)
            print(f"Processed {record_id}: {len(X_p)} segments.")
        except Exception as e:
            print(f"Error processing {record_id}: {e}")

    if all_X:
        print("Finalizing arrays and saving...")
        X_f = np.stack(all_X, axis=0) # [N, 2, 1000]
        y_f = np.stack(all_y, axis=0) # [N, 20]
        
        # Shuffle
        indices = np.random.permutation(len(X_f))
        split = int(len(X_f) * 0.8)
        
        train_idx, test_idx = indices[:split], indices[split:]
        
        torch.save({'X': torch.from_numpy(X_f[train_idx]), 
                    'y': torch.from_numpy(y_f[train_idx])}, 
                   os.path.join(SAVE_DIR, 'ltdb_train.pt'))
        
        torch.save({'X': torch.from_numpy(X_f[test_idx]), 
                    'y': torch.from_numpy(y_f[test_idx])}, 
                   os.path.join(SAVE_DIR, 'ltdb_test.pt'))
        
        print(f"Successfully saved {len(X_f)} segments to {SAVE_DIR}")