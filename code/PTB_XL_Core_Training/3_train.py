"""
PIPELINE STEP 3: Optimized Model Training & Evaluation (V3 Final Baseline)
Description:
    Final PTB-XL optimization using OneCycleLR, Label Smoothing, and 
    Class Weighting. Maintains original saving/loading structure.
"""
"""
================================================================================
PTB-XL OPTIMIZED TRAINING PIPELINE V3: CONFIGURATION & SEGMENTATION LOGIC
================================================================================

1. SEGMENTATION STRATEGY (2.5s Windows)
---------------------------------------
- Segment Size: 625 samples (2.5 seconds @ 250Hz).
- Transfer Learning Logic: This script is designed for downstream transfer 
  learning to the LTDB (MIT-BIH Long-Term ECG Database). 
- Rationale: 
  * LTDB utilizes beat-by-beat labeling. 
  * Using a full 10s record (2500 samples) "dilutes" the signal density of 
    transient arrhythmias. 
  * A 2.5s window ensures that a single abnormal beat significantly impacts 
    the segment's feature representation, preventing the "attenuation of 
    abnormality" that occurs in longer sequences.

2. HYPERPARAMETER EXPLAINER: "SMOOTHING": 0.05
----------------------------------------------
- What it does: Instead of forcing the model to predict exactly "1" (True) or 
  "0" (False), it tells the model to aim for "0.975" and "0.025".
- Why use it: 
  * Medical labeling is often "noisy" (different doctors might disagree on 
    borderline ECGs). 
  * It prevents the Transformer from becoming "overconfident." If the model is 
    too certain, it stops learning and begins to overfit. 
  * Smoothing encourages the model to learn the general "shape" of the 
    arrhythmia rather than memorizing specific training examples.

3. ARCHITECTURAL HPs
--------------------
- D_MODEL (128): Embedding width; balances feature depth with VRAM efficiency.
- N_HEADS (2): Allows the attention mechanism to look at the 'R-peak' and 
  'ST-segment' simultaneously.
- ONE-CYCLE LR: A "super-convergence" scheduler that starts slow, peaks, 
  and then cools down to find the flattest (most stable) local minima.
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Import the custom architecture defined in model.py
from model import LeadAgnosticTransformer

# ==========================================
# 1. Configuration & Path Management
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb')
OUTPUT_DIR = os.path.join(BASE_DIR,'..', '..', 'results', 'ptbxl_final')


def get_device():
    # Try to load DirectML for AMD GPUs
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        print("torch_directml not found. Checking for CUDA or CPU...")
        pass # Package not found, move on to the next check

    # Fall back to CUDA or CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



HP = {
    "LR_MAX": 4e-4,       # Peak learning rate for OneCycle; high enough for exploration, low enough for stability.
    "BATCH_SIZE": 32,    # Standard size to balance memory usage and gradient noise for better generalization.
    "EPOCHS": 60,         # Total iterations; OneCycleLR typically converges much faster than StepLR.
    "D_MODEL": 128,      # Embedding dimension; captures complex ECG features without excessive parameter count.
    "N_HEADS": 2,        # Multi-head attention count; allows model to focus on different ECG segments (e.g., P-wave vs QRS).
    "N_LAYERS": 2,       # Depth of Transformer; 2 layers is often sufficient for 1D signal classification.
    "DROPOUT": 0.3,      # Prevents overfitting by randomly deactivating neurons during training.
    "WEIGHT_DECAY": 1e-4, # L2 regularization for AdamW to keep weight magnitudes in check.
    "SMOOTHING": 0.05,    # Label smoothing; prevents overconfidence by softening target 0/1 values.
    "DEVICE": get_device(),
    "THRESHOLD": 0.5,    # Decision boundary for multi-label classification.
    "PATIENCE": 12,       # Early stopping buffer; allows the model to "recover" during the LR cooldown phase.
    "DELTA": 0.0001,     # Minimum improvement required to reset the early stopping counter.
    "MODEL_NAME": "best_model.pt"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Training Utilities
# ==========================================

class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def plot_performance(history, y_true, y_pred, class_names, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train')
    plt.plot(history["val_loss"], label='Val')
    plt.title('BCE Loss Trend')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["lr"], label='Learning Rate', color='orange')
    plt.title('OneCycle LR Schedule')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

# ==========================================
# 3. Main Execution Block
# ==========================================

def main():
    train_path = os.path.join(DATA_DIR, 'ptbxl_train_250.pt')
    test_path = os.path.join(DATA_DIR, 'ptbxl_test_250.pt')
    
    train_dict = torch.load(train_path, weights_only=False)
    test_dict = torch.load(test_path, weights_only=False)
    
    train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), batch_size=HP["BATCH_SIZE"])
    
    class_names = train_dict['classes']
    model = LeadAgnosticTransformer(
        num_classes=len(class_names),
        d_model=HP["D_MODEL"],
        nhead=HP["N_HEADS"],
        num_layers=HP["N_LAYERS"]
    ).to(HP["DEVICE"])
    
    # --- IMBALANCE HANDLING ---
    # Calculates weights to penalize the model more for missing rare classes (e.g., STTC, MI).
    counts = torch.tensor([1984, 1048, 2200, 3852, 2084], dtype=torch.float32)
    pos_weights = (counts.max() / counts).to(HP["DEVICE"])
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["LR_MAX"]/10, weight_decay=HP["WEIGHT_DECAY"])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=HP["LR_MAX"], 
        steps_per_epoch=len(train_loader), 
        epochs=HP["EPOCHS"]
    )

    early_stopping = EarlyStopping(
        patience=HP["PATIENCE"], 
        delta=HP["DELTA"], 
        path=os.path.join(OUTPUT_DIR, HP["MODEL_NAME"])
    )
    
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"Starting Final PTB-XL Optimization on {HP['DEVICE']}...")

    for epoch in range(HP["EPOCHS"]):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            # --- LABEL SMOOTHING LOGIC ---
            # Redistributes probability mass to reduce over-fitting to noisy binary labels.
            # Label Smoothing Logic
            y_smoothed = y * (1 - HP["SMOOTHING"]) + 0.5 * HP["SMOOTHING"]
            
            optimizer.zero_grad()
            out = model(x)
            # Use BCEWithLogitsLoss for multi-label tasks + class weights for imbalance.
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(out, y_smoothed)
            loss.backward()
            optimizer.step()
            scheduler.step() # Step every batch for OneCycle
            t_loss += loss.item()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                v_loss += nn.BCEWithLogitsLoss(pos_weight=pos_weights)(model(x.to(HP["DEVICE"])), y.to(HP["DEVICE"])).item()

        avg_train, avg_val = t_loss/len(train_loader), v_loss/len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(current_lr)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr:.6f}")
        
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final Evaluation (using your exact loading command)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, HP["MODEL_NAME"])))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(HP["DEVICE"]))
            all_preds.append((torch.sigmoid(out) > HP["THRESHOLD"]).cpu().numpy())
            all_labels.append(y.numpy())

    all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)
    plot_performance(history, all_labels, all_preds, class_names, OUTPUT_DIR)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("\nFinal Optimized PTB-XL Report:\n", report)

if __name__ == "__main__":
    main()