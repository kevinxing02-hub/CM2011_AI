import pandas as pd
import numpy as np
import wfdb
import os
import torch
import torch.nn as nn
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure your model.py has the Max Pooling + 4 Layers update
from model import LeadAgnosticTransformer

# ==========================================
# 1. Path Management
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'test_data', 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study')
# Update to your 10s model path
MODEL_PATH = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_10s_final', 'best_sequential_model.pt')

HP = {
    "TARGET_FS": 100,
    "WINDOW_SIZE": 1000,   # CHANGED: 10 seconds at 100Hz
    "BATCH_SIZE": 16,     # Reduced for 10s memory safety
    "NUM_CLASSES": 4,
    "MAX_BEATS": 20,      # MATCH: Must match your 10s training config
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

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
    
    record_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".hea"):
                record_paths.append(os.path.join(root, file[:-4]))

    if not record_paths:
        print("❌ No records found!")
        return None, None

    print(f"Found {len(record_paths)} records. Processing first 2000...")

    for full_record_path in record_paths[:2000]:
        try:
            signal, fields = wfdb.rdsamp(full_record_path)
            original_fs = fields['fs']
            
            # Resample to 100Hz (1000 samples for 10s)
            num_new = int(len(signal) * HP["TARGET_FS"] / original_fs)
            resampled_sig = resample(signal, num_new, axis=0)
            
            # Pad or Trim to exactly 1000 samples
            if len(resampled_sig) < HP["WINDOW_SIZE"]:
                resampled_sig = np.pad(resampled_sig, ((0, HP["WINDOW_SIZE"] - len(resampled_sig)), (0, 0)), mode='constant')
            else:
                resampled_sig = resampled_sig[:HP["WINDOW_SIZE"]]

            label = -1 
            for comment in fields['comments']:
                if 'Dx:' in comment:
                    dx_part = comment.split('Dx:')[1].strip()
                    codes = dx_part.split(',')
                    for c in codes:
                        if c.strip() in EXTERNAL_LABEL_MAP:
                            label = EXTERNAL_LABEL_MAP[c.strip()]
                            break
            
            if label == -1: continue 

            X_list.append(resampled_sig.T.astype(np.float32)) # [Leads, 1000]
            y_list.append(label) # Record-level label
                
        except Exception as e:
            print(f"Error processing {os.path.basename(full_record_path)}: {e}")

    return torch.tensor(np.array(X_list)), torch.tensor(np.array(y_list))

# ==========================================
# 3. Execution Logic
# ==========================================

if __name__ == "__main__":
    X_ext, y_ext = load_and_preprocess_chapman(DATA_DIR)
    
    # Initialize with 4 layers to match your new architecture
    model = LeadAgnosticTransformer(num_classes=5, num_layers=2).to(HP["DEVICE"])
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"]).to(HP["DEVICE"])
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=HP["DEVICE"]))
        print("✅ 10s Max-Pooling Model weights loaded.")
    
    model.eval()
    
    record_true = []
    record_pred = []

    print(f"Running 10s Record-Level inference...")
    with torch.no_grad():
        for i in range(0, len(X_ext), HP["BATCH_SIZE"]):
            x_batch = X_ext[i : i + HP["BATCH_SIZE"]].to(HP["DEVICE"])
            y_batch = y_ext[i : i + HP["BATCH_SIZE"]]
            
            # Z-score Normalization
            x_batch = (x_batch - x_batch.mean(dim=-1, keepdim=True)) / (x_batch.std(dim=-1, keepdim=True) + 1e-8)
            
            # Zero-out non-Lead II/V1 (Optional, but mimics your lead-agnostic setup)
            x_input = torch.zeros_like(x_batch)
            x_input[:, [1, 6], :] = x_batch[:, [1, 6], :]
            
            # Forward: [Batch, 80] -> [Batch, 20, 4]
            logits = model(x_input).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            probs = torch.softmax(logits, dim=2)
            
            for b_idx in range(len(probs)):
                # Get the predictions for all 20 beats in the 10s window
                beat_preds = torch.argmax(probs[b_idx], dim=1).cpu().numpy()
                
                # Record-Level Voting Logic:
                # If the model finds ANY Ventricular beat in the 10s, call the record Ventricular
                if 1 in beat_preds:
                    final_p = 1
                elif 2 in beat_preds:
                    final_p = 2
                elif 3 in beat_preds:
                    final_p = 3
                else:
                    final_p = 0
                
                record_pred.append(final_p)
                record_true.append(y_batch[b_idx].item())

    # 4. Final Report
    target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion']
    print("\n--- 10s RECORD-LEVEL GENERALIZATION (CHAPMAN) ---")
    print(classification_report(record_true, record_pred, labels=[0,1,2,3], target_names=target_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(record_true, record_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('10s External Validation Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(BASE_DIR, "external_validation_10s_cm.png"))
    print("✅ Confusion matrix saved.")