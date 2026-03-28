"""
PIPELINE STEP 4: LTDB Sequential Data Loading with Targeted Augmentation (FIXED)
"""
"""
================================================================================
PIPELINE STEP 4: LTDB SEQUENTIAL DATA PREPROCESSING & TARGETED AUGMENTATION
================================================================================

1. SEGMENTATION STRATEGY: 2.5s HIGH-DENSITY WINDOWS
---------------------------------------------------
- WINDOW SIZE: 250 samples (2.5s @ 100Hz). 
- DENSITY PRINCIPLE: As established in LTDB benchmarks, 10s sequences "suck" 
  due to the attenuation of pathology density. By using a 2.5s window, we 
  ensure that transient arrhythmias (rare beats) dominate the temporal segment 
  rather than being "washed out" by surrounding normal sinus rhythm.

2. BASEMENT PRINCIPLE: THE 4-CLASS AAMI MAPPING
-----------------------------------------------
- RATIONALE: We map diverse clinical symbols (N, L, R, V, E, A, a, J, S, F, f, Q) 
  into 4 unified subclasses based on the AAMI (Association for the Advancement 
  of Medical Instrumentation) standard.
- CLASS 0 (Normal/Bundle Branch): Standard rhythm baseline.
- CLASS 1 (Ventricular): Critical arrhythmias (PVC/E).
- CLASS 2 (Supraventricular): Ectopic beats (PAC/SVT).
- CLASS 3 (Fusion/Unknown): Complex morphological overlaps.
- WHY 4 CLASSES?: This standardizes the "medical vocabulary" between different 
  datasets (PTB-XL, LTDB, and Chapman), enabling seamless Transfer Learning 
  and highly interpretable clinical results.

3. TARGETED AUGMENTATION FOR RARE PATHOLOGIES
---------------------------------------------
- THE PROBLEM: LTDB is heavily imbalanced; Normal beats (Class 0) overwhelm 
  Rare Classes (2 & 3). 
- THE STRATEGY: We perform 'Targeted Jittering.' When a Rare Class (S or F) 
  is detected, we do not just take one window. Instead, we generate 3 windows:
    1. Centered on the beat.
    2. Offset by -0.2s (shifting the beat to the right).
    3. Offset by +0.2s (shifting the beat to the left).
- RESULT: This triples the representation of rare pathologies in the training 
  set, forcing the Transformer to learn morphological features regardless of 
  where the beat appears in the 2.5s window.

4. SEQUENTIAL LABEL PADDING
---------------------------
- We use a "MAX_BEATS" (6) padding strategy. If a 2.5s window contains only 
  3 beats, the label is padded with '-1' (ignored in loss). This allows the 
  model to output a fixed-length sequence while handling variable heart rates.
================================================================================
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