"""
PIPELINE STEP 5: Sequential Transfer Learning (Experiment v1-Refined)
Description:
    Uses Focal Loss to focus on hard samples without explicit weight passing.
    Includes full visualization and early stopping to match v2 structure.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

from model import LeadAgnosticTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_train.pt'))
TEST_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_test.pt'))
PRETRAINED_PATH = os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_v1', 'best_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v3_v1refined')

HP = {
    "BATCH_SIZE": 32,
    "EPOCHS": 38,
    "PATIENCE": 8,
    "LR_ENCODER": 3e-5,
    "LR_HEAD": 1e-3,
    "MAX_BEATS": 6,
    "NUM_CLASSES": 4, 
    "GAMMA": 1.5,  # Focal Loss focusing parameter
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class FocalLoss(nn.Module):
    """Special Loss Function: Focusing on hard-to-classify beats."""
    def __init__(self, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        logpt = self.ce(inputs, targets)
        pt = torch.exp(-logpt)
        # The 'focal' part: (1-pt)^gamma effectively ignores easy (high pt) samples
        loss = ((1 - pt) ** self.gamma) * logpt
        return loss.mean()

def get_sequential_model():
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=HP["DEVICE"]))
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
    return model.to(HP["DEVICE"])

def run_transfer_session():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_dict = torch.load(TRAIN_DATA_PATH, weights_only=False)
    test_dict = torch.load(TEST_DATA_PATH, weights_only=False)
    
    train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), batch_size=HP["BATCH_SIZE"])
    
    model = get_sequential_model()
    # Using Focal Loss instead of Weighted Cross Entropy
    criterion = FocalLoss(gamma=HP["GAMMA"], ignore_index=-1)
    
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if "classifier" not in n], 'lr': HP["LR_ENCODER"]},
        {'params': model.classifier.parameters(), 'lr': HP["LR_HEAD"]}
    ])

    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "f1": []}

    print(f"Starting Refined V1 Training (Focal Loss, Gamma={HP['GAMMA']})...")

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

        model.eval()
        total_val_loss = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x_g, y_g = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
                logits = model(x_g).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
                v_loss = criterion(logits.transpose(1, 2), y_g)
                total_val_loss += v_loss.item()
                
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                targets = y.numpy()
                for i in range(len(targets)):
                    for j in range(HP["MAX_BEATS"]):
                        if targets[i, j] != -1:
                            all_preds.append(preds[i, j])
                            all_true.append(targets[i, j])

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)
        report_dict = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["f1"].append(macro_f1)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f" --> Best Model Saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["PATIENCE"]:
                print("Early stopping triggered.")
                break

    # Final Evaluation & Visualization
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.eval()


    # --- FINAL EVALUATION & VISUALIZATION ---
    print("\n--- GENERATING FINAL REPORT & VISUALS ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.eval()
    
    final_preds, final_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x_gpu, y_gpu = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            logits = model(x_gpu).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            targets = y.numpy()
            
            for i in range(len(targets)):
                for j in range(HP["MAX_BEATS"]):
                    if targets[i, j] != -1:  # Ignore padding
                        final_preds.append(preds[i, j])
                        final_true.append(targets[i, j])

    # 1. Save Classification Report
    target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion/Other']
    report = classification_report(final_true, final_preds, target_names=target_names, zero_division=0)
    print(report)
    
    report_path = os.path.join(OUTPUT_DIR, "sequential_report.txt")
    with open(report_path, "w") as f:
        f.write("LTDB SEQUENTIAL TRANSFER LEARNING (FOCAL LOSS) REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)

    # 2. Save Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train Focal Loss')
    plt.plot(history["val_loss"], label='Val Focal Loss')
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

    # 3. Save Confusion Matrix
    cm = confusion_matrix(final_true, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Focal Loss Experiment')
    plt.ylabel('Actual Beat')
    plt.xlabel('Predicted Beat')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Results saved successfully to: {OUTPUT_DIR}")
    # [Insert Confusion Matrix and classification_report logic here as in v2]
    # (Same as v2 logic to save confusion_matrix.png and sequential_report.txt)

if __name__ == "__main__":
    run_transfer_session()