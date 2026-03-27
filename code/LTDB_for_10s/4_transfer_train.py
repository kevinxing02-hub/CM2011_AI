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
# Updated paths to reflect the '10' or '10s' naming convention
TRAIN_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb_10s', 'ltdb_train.pt'))
TEST_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb_10s', 'ltdb_test.pt'))
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_10s_full', 'best_10s_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_10s_final')

HP = {
    "BATCH_SIZE": 16,    # REDUCED from 32 to prevent Out-of-Memory (OOM) on 10s sequences
    "EPOCHS": 150,
    "PATIENCE": 20,      # Slightly more patience for longer sequences
    "LR_ENCODER": 3e-5,  # Lowered slightly for stable fine-tuning
    "LR_HEAD": 1e-3,
    "MAX_BEATS": 20,     # MATCH: Must match CONFIG["max_beats"] in 4_ltdb_loader.py
    "NUM_CLASSES": 4, 
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_sequential_model():
    """Adjusts the LeadAgnosticTransformer for sequential beat prediction."""
    # CHANGE: Add num_classes=5 to match the pre-trained PTB-XL architecture
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    
    if os.path.exists(PRETRAINED_WEIGHTS):
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=HP["DEVICE"]))
        print(f"Loaded 10s pre-trained weights from {PRETRAINED_WEIGHTS}")
    
    # Now replace the head for LTDB Sequential [MAX_BEATS * NUM_CLASSES]
    model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
    return model.to(HP["DEVICE"])

def run_transfer_session():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {TRAIN_DATA_PATH}...")
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
    # ignore_index=-1 skips the padding tokens in the loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 3. Training Loop
    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "f1": []}

    print(f"Starting 10s Transfer Training (Max Epochs: {HP['EPOCHS']})...")

    for epoch in range(HP["EPOCHS"]):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            optimizer.zero_grad()
            
            # Forward pass: shape [Batch, MAX_BEATS * NUM_CLASSES]
            logits = model(x).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            
            # CrossEntropy expects [Batch, Classes, Seq_Len]
            loss = criterion(logits.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation
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
                        if targets[i, j] != -1: # Only evaluate real beats
                            all_preds.append(preds[i, j])
                            all_true.append(targets[i, j])

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)
        
        report_dict = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["f1"].append(macro_f1)

        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sequential_model.pt"))
            print(" --> Best Model Saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["PATIENCE"]:
                print(f"Early stopping triggered.")
                break

    # 4. Final Evaluation & Visuals
    print("\n--- GENERATING FINAL 10s REPORT ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_sequential_model.pt")))
    model.eval()
    
    final_preds, final_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(HP["DEVICE"])).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            targets = y.numpy()
            for i in range(len(targets)):
                for j in range(HP["MAX_BEATS"]):
                    if targets[i, j] != -1:
                        final_preds.append(preds[i, j])
                        final_true.append(targets[i, j])

    target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion/Other']
    report = classification_report(final_true, final_preds, target_names=target_names, zero_division=0)
    print(report)
    
    # (Plotting logic remains the same as your previous script)
    # ... [Insert your plotting code here] ...

if __name__ == "__main__":
    run_transfer_session()