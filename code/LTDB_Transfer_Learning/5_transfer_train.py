"""
================================================================================
LTDB SEQUENTIAL TRANSFER LEARNING: BEAT-BY-BEAT CLASSIFICATION (V1)
================================================================================

1. SEGMENTATION STRATEGY: THE 2.5s VS. 10s BENCHMARK !!
---------------------------------------------------------------
- RATIONALE: During LTDB testing, 2.5s windows (625 samples) significantly 
  outperform 10s windows. 
- WHY 10s FAILS: Using a 10s sequence "sucks" because it causes a severe 
  ATTENUATED DENSITY of pathological signals. In a 10s strip, a single 
  arrhythmic beat is diluted by 9 seconds of background noise/normal beats.
- THE 2.5s ADVANTAGE: Shortening to 2.5s increases the relative density of the 
  abnormality within the window. This ensures the Transformer's attention 
  mechanism isn't "overwhelmed" by normal sinus rhythm, allowing it to 
  capture transient arrhythmic events with much higher sensitivity.

2. HYPERPARAMETER EXPLAINER (FOR PEER REVIEW)
---------------------------------------------
- "MAX_BEATS" (6): Caps the sequence length at 6 QRS complexes. This matches 
  the high-density 2.5s window logic, ensuring a tight mapping between 
  signal segments and beat-by-beat labels.
- "LR_ENCODER" (5e-5) vs "LR_HEAD" (1e-3): Differential learning rates. 
  We "freeze" the PTB-XL feature extraction logic with a low LR while 
  allowing the new Sequential Head to learn LTDB mapping rapidly.
- "SMOOTHING" (0.05): Prevents over-fitting to specific "hard" labels, 
  encouraging the model to learn the general morphological "spirit" of the 
  arrhythmia across different datasets.

3. EXECUTION PROCESS
--------------------
- Data: Sequential tensors [X: Signals, y: Beat Labels].
- Loss: CrossEntropy with 'ignore_index=-1' to handle variable beat counts.
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

# Ensure the path to model.py is accessible
from model import LeadAgnosticTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_train.pt'))
TEST_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_test.pt'))
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_final', 'best_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v1_ultimate_F')

def get_device():
    # Try to load DirectML for AMD GPUs
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass # Package not found, move on to the next check

    # Fall back to CUDA or CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

HP = {
    "BATCH_SIZE": 32,    # Standard power-of-2 size; balances VRAM and gradient stability.
    "EPOCHS": 150,       # High limit; the model will likely stop early via the Patience monitor.
    "PATIENCE": 18,      # Early stopping buffer; long enough to allow the head to adapt to the encoder.
    
    # --- DIFFERENTIAL LEARNING RATES ---
    "LR_ENCODER": 5e-5,  # 10-20x smaller than head; prevents "catastrophic forgetting" of PTB-XL features.
    "LR_HEAD": 1e-3,     # High LR for the new classifier to quickly learn LTDB-specific beat mapping.
    
    # --- SEQUENTIAL MAPPING ---
    "MAX_BEATS": 6,      # Max number of QRS complexes in a 2.5s window (covers HR up to 144 BPM).
    "NUM_CLASSES": 4,    # Target classes: Normal, Ventricular, Supraventricular, Fusion/Other.
    "DEVICE": get_device()
}

def get_sequential_model():
    """
    TRANSFORMATION LOGIC:
    1. Loads the LeadAgnosticTransformer (5-class PTB-XL version).
    2. Injects pre-trained weights into the Transformer layers.
    3. Replaces the final classifier with a flattened Sequential Head 
       of size [Max_Beats * Num_Classes] to output beat-by-beat labels.
    """
    """Adjusts the LeadAgnosticTransformer for sequential beat prediction."""
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    if os.path.exists(PRETRAINED_WEIGHTS):
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=HP["DEVICE"]))
        print(f"Loaded pre-trained weights from {PRETRAINED_WEIGHTS}")
    
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
    return model.to(HP["DEVICE"])

def run_transfer_session():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    train_dict = torch.load(TRAIN_DATA_PATH, weights_only=False)
    test_dict = torch.load(TEST_DATA_PATH, weights_only=False)
    
    train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), 
                              batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), 
                             batch_size=HP["BATCH_SIZE"])
    
    model = get_sequential_model()
    
    # 2. Optimizer & Loss
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if "classifier" not in n], 'lr': HP["LR_ENCODER"]},
        {'params': model.classifier.parameters(), 'lr': HP["LR_HEAD"]}
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 3. Training Loop with Early Stopping
    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "f1": []}

    print(f"Starting V1 Training (CrossEntropy, Max Epochs: {HP['EPOCHS']})...")

    for epoch in range(HP["EPOCHS"]):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            optimizer.zero_grad()
            logits = model(x).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            loss = criterion(logits.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation & Metrics Collection
        model.eval()
        total_val_loss = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x_g, y_g = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
                logits = model(x_g).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
                val_loss = criterion(logits.transpose(1, 2), y_g)
                total_val_loss += val_loss.item()
                
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                targets = y.numpy()
                for i in range(len(targets)):
                    for j in range(HP["MAX_BEATS"]):
                        if targets[i, j] != -1:
                            all_preds.append(preds[i, j])
                            all_true.append(targets[i, j])

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)
        
        # Calculate Macro F1 for Early Stopping
        report_dict = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["f1"].append(macro_f1)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Macro F1: {macro_f1:.4f}")

        # Early Stopping Logic (Monitoring Macro F1)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sequential_model.pt"))
            print(" --> Best Model Saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["PATIENCE"]:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 4. Final Evaluation & Visualization
    print("\n--- GENERATING FINAL REPORT & VISUALS ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_sequential_model.pt")))
    model.eval()
    
    final_preds, final_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_gpu = x.to(HP["DEVICE"])
            logits = model(x_gpu).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            targets = y.numpy()
            
            for i in range(len(targets)):
                for j in range(HP["MAX_BEATS"]):
                    if targets[i, j] != -1:
                        final_preds.append(preds[i, j])
                        final_true.append(targets[i, j])

    # Save Text Report
    target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion/Other']
    report = classification_report(final_true, final_preds, target_names=target_names, zero_division=0)
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, "sequential_report.txt"), "w") as f:
        f.write("LTDB SEQUENTIAL TRANSFER LEARNING (V1 BASELINE) REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)

    # Save Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["f1"], label='Macro F1', color='green')
    plt.title('Validation Macro F1')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"))
    plt.close()

    # Save Confusion Matrix
    cm = confusion_matrix(final_true, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - V1 Baseline')
    plt.ylabel('Actual Beat')
    plt.xlabel('Predicted Beat')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_transfer_session()