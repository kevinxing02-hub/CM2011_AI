


import pandas as pd
import numpy as np
import wfdb
import os
import torch
import torch.nn as nn
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Ensure your model.py is in the same folder or PYTHONPATH
from model import LeadAgnosticTransformer

# ==========================================
# 1. Path Management (Updated for your structure)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your downloaded Chapman dataset
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'test_data', 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study')
# Path to your best sequential model
MODEL_PATH = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_final', 'best_sequential_model.pt')

HP = {
    "TARGET_FS": 100,      #
    "WINDOW_SIZE": 250,    #
    "BATCH_SIZE": 32,
    "NUM_CLASSES": 4,      #
    "MAX_BEATS": 6,        #
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Mapping SNOMED codes from your .csv to AAMI
EXTERNAL_LABEL_MAP = {
    '426783006': 0, # NSR
    '164884008': 1, # PVC -> Ventricular
    '284470004': 1, # R-on-T PVC -> Ventricular
    '63593006':  2, # PAC -> Supraventricular
    '426434006': 2, # SVT -> Supraventricular
}
def load_and_preprocess_chapman(base_path):
    print(f"Searching for records in: {base_path}")
    X_list, y_list = [], []
    
    # 1. Walk through all subdirectories to find .hea files
    record_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".hea"):
                # wfdb needs the path without the extension
                record_paths.append(os.path.join(root, file[:-4]))

    if not record_paths:
        print("❌ No records found! Check if DATA_DIR is correct.")
        return torch.tensor([]), torch.tensor([])

    print(f"Found {len(record_paths)} records. Processing first 500...")

    # 2. Process records
    #for full_record_path in record_paths[:500]:
        # Change from 500 to 5000
    for full_record_path in record_paths[:5000]:
        try:
            # Load 500Hz signal
            signal, fields = wfdb.rdsamp(full_record_path)
            original_fs = fields['fs']
            
            # Resample to 100Hz
            num_new = int(len(signal) * HP["TARGET_FS"] / original_fs)
            resampled_sig = resample(signal, num_new, axis=0)
            
            # Extract Label from Header
            label = -1 
            for comment in fields['comments']:
                if 'Dx:' in comment:
                    # Some headers use 'Dx: code1,code2'
                    dx_part = comment.split('Dx:')[1].strip()
                    codes = dx_part.split(',')
                    for c in codes:
                        clean_code = c.strip()
                        if clean_code in EXTERNAL_LABEL_MAP:
                            label = EXTERNAL_LABEL_MAP[clean_code]
                            break
            
            if label == -1: 
                continue # Skip if the disease isn't in our N, V, S, F mapping

            # Segment into 2.5s chunks
            for start in range(0, len(resampled_sig) - HP["WINDOW_SIZE"], HP["WINDOW_SIZE"]):
                end = start + HP["WINDOW_SIZE"]
                seg = resampled_sig[start:end].T # [Leads, Time]
                
                # Create sequential label [Label, -1, -1, -1, -1, -1]
                seq_y = [label] + [-1] * (HP["MAX_BEATS"] - 1)
                
                X_list.append(seg.astype(np.float32))
                y_list.append(np.array(seq_y, dtype=np.int64))
                
        except Exception as e:
            # Log specific errors for debugging
            print(f"Error processing {os.path.basename(full_record_path)}: {e}")

    if not X_list:
        print("❌ All records skipped! Check EXTERNAL_LABEL_MAP codes.")
        return torch.tensor([]), torch.tensor([])

    return torch.tensor(np.array(X_list)), torch.tensor(np.array(y_list))

# ==========================================
# 3. Execution Logic
# ==========================================

if __name__ == "__main__":
    # 1. Load Data
    X_ext, y_ext = load_and_preprocess_chapman(DATA_DIR)
    
    # 2. Setup Evaluator
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    # Reconstruct transfer head
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"]).to(HP["DEVICE"])
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=HP["DEVICE"]))
        print("✅ Model weights loaded.")
    else:
        print(f"❌ Could not find model at {MODEL_PATH}")

    model.eval()
    
    # 3. Predict
    # 3. Predict with Record-Level Aggregation
    loader = DataLoader(TensorDataset(X_ext, y_ext), batch_size=HP["BATCH_SIZE"])
    
    # We will group by 'record' instead of individual 2.5s segments
    record_true = []
    record_pred = []

    print(f"Running Record-Level inference...")
    with torch.no_grad():
        # Process in chunks of 4 (since each 10s record produced 4 segments)
        for i in range(0, len(X_ext), 4):
            x_batch = X_ext[i:i+4].to(HP["DEVICE"])
            y_batch = y_ext[i:i+4]
            
            # Normalization
            x_batch = (x_batch - x_batch.mean(dim=-1, keepdim=True)) / (x_batch.std(dim=-1, keepdim=True) + 1e-8)
            
            # Lead selection
            x_input = torch.zeros_like(x_batch)
            x_input[:, [1, 6], :] = x_batch[:, [1, 6], :]
            
            logits = model(x_input).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            preds = torch.argmax(logits, dim=2).cpu().numpy() # [4, 6]
            
            # The Logic: If the record is labeled '1' (Ventricular), 
            # did the model find A VENTRICULAR BEAT ANYWHERE in the 4 segments?
            actual_label = y_batch[0, 0].item()
            predicted_as_v = 1 if (preds == 1).any() else 0
            
            record_true.append(actual_label)
            record_pred.append(predicted_as_v if actual_label == 1 else preds[0,0])

    # 4. Final Report
    target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion']
    print("\n--- RECORD-LEVEL GENERALIZATION (CHAPMAN) ---")
    print(classification_report(record_true, record_pred, labels=[0,1,2,3], target_names=target_names, zero_division=0))

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Generate the CM
    cm = confusion_matrix(record_true, record_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('External Validation Confusion Matrix (Chapman-Shaoxing)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(BASE_DIR, "external_validation_cm.png"))
    print("✅ Confusion matrix saved to external_validation_cm.png")