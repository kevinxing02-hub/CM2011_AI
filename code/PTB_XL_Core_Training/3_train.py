"""
PIPELINE STEP 3: Model Training & Evaluation
Description:
    Loads segmented ECG data, trains the LeadAgnosticTransformer, 
    and saves performance metrics/visualizations.
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
# Data directory aligns with split_data.py output
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb')
OUTPUT_DIR = os.path.join(BASE_DIR,'..', '..', 'results', 'ptbxl_v1')


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
    "LR": 1e-5,
    "BATCH_SIZE": 32,
    "EPOCHS": 50,         
    "D_MODEL": 128,      
    "N_HEADS": 2,
    "N_LAYERS": 2,
    "DROPOUT": 0.2,
    "DEVICE": get_device(),
    "THRESHOLD": 0.5, # Threshold for multi-label classification
    "PATIENCE": 10,   # For early stopping
    "DELTA": 0.0001,  
    "MODEL_NAME": "best_model.pt"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Training Utilities
# ==========================================

class EarlyStopping:
    """Monitors validation loss and stops training to prevent overfitting."""
    def __init__(self, patience=7, delta=0, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def plot_performance(history, y_true, y_pred, class_names, output_dir):
    """Generates loss curves and confusion matrices for all classes."""
    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label='Train')
    plt.plot(history["val_loss"], label='Val')
    plt.title('BCE Loss Trend')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Confusion Matrices
    cms = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    
    if num_classes == 1: axes = [axes]
    
    for i, (cm, name) in enumerate(zip(cms, class_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Class: {name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

# ==========================================
# 3. Main Execution Block
# ==========================================

def main():
    # Load the 2.5s segmented data produced by split_data.py
    train_path = os.path.join(DATA_DIR, 'ptbxl_train_250.pt')
    test_path = os.path.join(DATA_DIR, 'ptbxl_test_250.pt')
    
    if not os.path.exists(train_path):
        print("Data files missing. Ensure data_loader.py and split_data.py were run.")
        return

    train_dict = torch.load(train_path, weights_only=False)
    test_dict = torch.load(test_path, weights_only=False)
    
    train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), batch_size=HP["BATCH_SIZE"])
    
    # Initialize Model
    class_names = train_dict['classes']
    model = LeadAgnosticTransformer(
        num_classes=len(class_names),
        d_model=HP["D_MODEL"],
        nhead=HP["N_HEADS"],
        num_layers=HP["N_LAYERS"]
    ).to(HP["DEVICE"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=HP["LR"])
    criterion = nn.BCEWithLogitsLoss() # Suitable for multi-label classification
    
    early_stopping = EarlyStopping(
        patience=HP["PATIENCE"], 
        delta=HP["DELTA"], 
        path=os.path.join(OUTPUT_DIR, HP["MODEL_NAME"])
    )
    
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on: {HP['DEVICE']}")
    for epoch in range(HP["EPOCHS"]):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                v_loss += criterion(model(x.to(HP["DEVICE"])), y.to(HP["DEVICE"])).item()

        avg_train, avg_val = t_loss/len(train_loader), v_loss/len(test_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f}")
        
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Evaluation with best weights
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
    
    # Save text report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    main()