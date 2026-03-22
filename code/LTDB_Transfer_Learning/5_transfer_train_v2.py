"""
PIPELINE STEP 5: Sequential Transfer Learning with Early Stopping
Description:
    Includes weighted loss, automated reporting, and an early stopping 
    mechanism based on Macro F1 to prevent overfitting.
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
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v2')

HP = {
    "BATCH_SIZE": 32,
    "EPOCHS": 20,              # Increased max epochs since early stopping will catch the end
    "PATIENCE": 3,             # Number of epochs to wait for improvement before stopping
    "LR_ENCODER": 3e-5,
    "LR_HEAD": 1e-3,
    "MAX_BEATS": 6,
    "NUM_CLASSES": 4, 
    "DEVICE": torch.device("cuda" if torch.device("cuda" if torch.cuda.is_available() else "cpu") else "cpu")
}

def calculate_class_weights(labels):
    flat_labels = labels[labels != -1].numpy()
    counts = np.bincount(flat_labels.astype(int), minlength=HP["NUM_CLASSES"])
    weights = len(flat_labels) / (HP["NUM_CLASSES"] * counts)
    return torch.tensor(weights, dtype=torch.float32).to(HP["DEVICE"])

def run_transfer_session():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    train_dict = torch.load(TRAIN_DATA_PATH, weights_only=False)
    test_dict = torch.load(TEST_DATA_PATH, weights_only=False)
    class_weights = calculate_class_weights(train_dict['y'])
    
    train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), batch_size=HP["BATCH_SIZE"])
    
    # 2. Model & Optimization
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=HP["DEVICE"]))
    
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
    model.to(HP["DEVICE"])

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if "classifier" not in n], 'lr': HP["LR_ENCODER"]},
        {'params': model.classifier.parameters(), 'lr': HP["LR_HEAD"]}
    ])

    # 3. Training with Early Stopping Logic
    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "f1": []}

    print(f"Starting Training with Early Stopping (Patience: {HP['PATIENCE']})...")

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

        # Evaluation phase
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

        report_dict = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Macro F1: {macro_f1:.4f}")

        # Early Stopping Check
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f" --> New Best Model Saved (F1: {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["PATIENCE"]:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 4. Final Reporting (Load best weights)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    # ... [Insert Confusion Matrix and Report generation logic from previous script here]

if __name__ == "__main__":
    run_transfer_session()