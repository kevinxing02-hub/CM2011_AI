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
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_v1', 'best_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v1_ultimate_2')

HP = {
    "BATCH_SIZE": 32,
    "EPOCHS": 150,
    "PATIENCE": 11,      # Added: Early stopping patience
    "LR_ENCODER": 5e-5,
    "LR_HEAD": 1e-3,
    "MAX_BEATS": 6,  
    "NUM_CLASSES": 4, 
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_sequential_model():
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