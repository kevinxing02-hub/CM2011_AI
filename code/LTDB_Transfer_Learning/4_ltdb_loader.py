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
    "window_seconds": 2.5,
    "stride_seconds": 1.75,
    "max_beats": 6,
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
RAW_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'raw_data', 'ltdb'))
SAVE_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb')

def get_padded_label(window_labels):
    if len(window_labels) < CONFIG["max_beats"]:
        return window_labels + [-1] * (CONFIG["max_beats"] - len(window_labels))
    return window_labels[:CONFIG["max_beats"]]

def process_ltdb_signals(record_name, data_dir):
    path = os.path.join(data_dir, record_name)
    signal, fields = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, 'atr')
    original_fs = fields['fs']
    
    num_samples_new = int(len(signal) * CONFIG["target_fs"] / original_fs)
    resampled_signal = resample(signal, num_samples_new)
    
    win_size = int(CONFIG["window_seconds"] * CONFIG["target_fs"])
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
            seg = resampled_signal[start:end].T
            if seg.shape == (2, 250): # Strict shape check
                X_list.append(seg.astype(np.float32))
                y_list.append(np.array(get_padded_label(labels), dtype=np.int64))
                processed_starts.add(start)

    # --- 2. Targeted Augmentation ---
    for i, symbol in enumerate(ann.symbol):
        label = LABEL_MAP.get(symbol)
        if label in RARE_CLASSES:
            for offset_sec in [0] + CONFIG["aug_offsets"]:
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
                    if seg.shape == (2, 250):
                        X_list.append(seg.astype(np.float32))
                        y_list.append(np.array(get_padded_label(labels), dtype=np.int64))
                        processed_starts.add(start)

    return X_list, y_list


def count_ltdb_annotations(records, data_dir):
    overall_counts = Counter()
    
    print(f"Starting Annotation Count for {len(records)} records")
    
    for record in records:
        try:
            path = os.path.join(data_dir, record)
            # We only need the annotations ('atr') for the count
            ann = wfdb.rdann(path, 'atr')
            
            # Update the counter with the labels from this specific record
            overall_counts.update(ann.symbol)
            
        except Exception as e:
            print(f" Error reading annotations for {record}: {e}")
            
    return overall_counts

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    record_paths = glob.glob(os.path.join(RAW_DATA_PATH, "*.hea"))
    records = [os.path.basename(f).replace('.hea', '') for f in record_paths]


    annotation_results = count_ltdb_annotations(records, RAW_DATA_PATH)

    print("\nAnnotation Totals ")
    # Sorting by frequency (highest first)
    for symbol, count in annotation_results.most_common():
        print(f"Annotation '{symbol}': {count}")

    all_X, all_y = [], []
    for record in records:
        try:
            X_p, y_p = process_ltdb_signals(record, RAW_DATA_PATH)
            all_X.extend(X_p)
            all_y.extend(y_p)
            print(f"Processed {record}: {len(X_p)} segments.")
        except Exception as e:
            print(f"Error {record}: {e}")

    if all_X:
        # THE FIX: Explicitly stack into a 3D array for X and 2D for y
        print("Finalizing arrays... (This fixes the Inhomogeneous Shape error)")
        X_f = np.stack(all_X, axis=0) # Result: [N, 2, 250]
        y_f = np.stack(all_y, axis=0) # Result: [N, 6]
        
        indices = np.random.permutation(len(X_f))
        split = int(len(X_f) * 0.8)
        
        train_idx, test_idx = indices[:split], indices[split:]
        
        torch.save({'X': torch.from_numpy(X_f[train_idx]), 
                    'y': torch.from_numpy(y_f[train_idx])}, 
                   os.path.join(SAVE_DIR, 'ltdb_train.pt'))
        
        torch.save({'X': torch.from_numpy(X_f[test_idx]), 
                    'y': torch.from_numpy(y_f[test_idx])}, 
                   os.path.join(SAVE_DIR, 'ltdb_test.pt'))
        
        print(f"Successfully saved {len(X_f)} segments.")