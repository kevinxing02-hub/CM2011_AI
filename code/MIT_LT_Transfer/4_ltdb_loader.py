"""
PIPELINE STEP 4: LTDB Data Loading & Transfer Preparation
Description: 
    Processes Long-Term ECG Database (LTDB) records into 2.5s segments 
    to match the PTB-XL pre-training window.
"""
import numpy as np
import wfdb
import os
import torch
import glob
from scipy.signal import resample
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

CONFIG = {
    "target_fs": 100,                 # Must match PTB-XL
    "window_seconds": 2.5,            # Must match 250 samples
    "stride_seconds": 1.75,           # Overlap for data augmentation
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'raw_data', 'ltdb'))
SAVE_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb')

def process_ltdb_signals(record_name, data_dir):
    path = os.path.join(data_dir, record_name)
    signal, fields = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, 'atr')
    original_fs = fields['fs']
    
    # Resample to 100Hz
    num_samples_new = int(len(signal) * CONFIG["target_fs"] / original_fs)
    resampled_signal = resample(signal, num_samples_new)
    
    window_size = int(CONFIG["window_seconds"] * CONFIG["target_fs"]) # 250 samples
    stride = int(CONFIG["stride_seconds"] * CONFIG["target_fs"])

    X_list, y_list = [], []
    
    for start in range(0, len(resampled_signal) - window_size, stride):
        end = start + window_size
        orig_start = int(start * original_fs / CONFIG["target_fs"])
        orig_end = int(end * original_fs / CONFIG["target_fs"])
        
        indices = np.where((ann.sample >= orig_start) & (ann.sample < orig_end))[0]
        window_anns = [ann.symbol[i] for i in indices]
        
        if len(window_anns) > 0:
            counts = Counter(window_anns)
            label, count = counts.most_common(1)[0]
            
            # Purity check: Arrhythmias allowed at 30% majority (per your code logic); Normal must be 100%
            is_pure = len(set(window_anns)) == 1
            is_majority_arr = (label != 'N') and (count / len(window_anns) >= 0.3)
            
            if (is_pure or is_majority_arr) and label not in ['~', '|', '+', 'x']:
                segment = resampled_signal[start:end].T # [Leads, Time]
                if segment.shape == (2, 250): # LTDB typically has 2 leads
                    X_list.append(segment)
                    y_list.append([label])

    return X_list, y_list

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    raw_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.hea"))
    records = [os.path.basename(f).replace('.hea', '') for f in raw_files]

    print(f"--- Starting LTDB Processing ---")
    print(f"Found {len(records)} records in {RAW_DATA_PATH}")

    all_X, all_y = [], []
    for record in records:
        try:
            X_p, y_p = process_ltdb_signals(record, RAW_DATA_PATH)
            all_X.extend(X_p)
            all_y.extend(y_p)
            print(f" Successfully processed {record}: {len(X_p)} segments extracted.")
        except Exception as e:
            print(f" Error processing {record}: {e}")

    if len(all_X) == 0:
        print("\n[!] ERROR: No segments were found. Please check your data paths or 'purity' logic.")
    else:
        X_final = np.array(all_X)
        mlb = MultiLabelBinarizer()
        y_final = mlb.fit_transform(all_y)

        # 80/20 Train/Test Split
        indices = np.random.permutation(len(X_final))
        split_idx = int(len(X_final) * 0.8)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        # Saving
        torch.save({'X': torch.tensor(X_final[train_idx], dtype=torch.float32),
                    'y': torch.tensor(y_final[train_idx], dtype=torch.float32),
                    'classes': mlb.classes_}, 
                   os.path.join(SAVE_DIR, 'ltdb_train.pt'))
        
        torch.save({'X': torch.tensor(X_final[test_idx], dtype=torch.float32),
                    'y': torch.tensor(y_final[test_idx], dtype=torch.float32),
                    'classes': mlb.classes_}, 
                   os.path.join(SAVE_DIR, 'ltdb_test.pt'))

        # --- NEW VISUALIZATION OUTPUTS ---
        print("\n" + "="*40)
        print("PROCESSING SUMMARY")
        print("="*40)
        print(f"Total Segments Extracted: {len(X_final)}")
        print(f"Training Segments:      {len(train_idx)}")
        print(f"Testing Segments:       {len(test_idx)}")
        print(f"Data Shape:             {X_final.shape}")
        print("-" * 40)
        
        # Flatten all_y to get counts of each individual label
        label_counts = Counter([label for sublist in all_y for label in sublist])
        print("CLASS DISTRIBUTION:")
        for cls, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_y)) * 100
            print(f"  [{cls}]: {count:>6} samples ({percentage:>5.2f}%)")
        
        print("-" * 40)
        print(f"Files saved to: {SAVE_DIR}")
        print("="*40)