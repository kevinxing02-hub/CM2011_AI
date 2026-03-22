"""
PIPELINE STEP 5: Sequential Transfer Learning (v1 + Early Stopping + LR Scheduler)
Description:
    Fine-tunes the pre-trained PTB-XL Lead-Agnostic Transformer 
    to predict a sequence of beat labels for LTDB segments.
    Includes Early Stopping, Learning Rate Scheduling, and Visual Reporting.
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
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_v1', 'best_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v1_plus')

HP = {
    "BATCH_SIZE": 32,
    "EPOCHS": 150,           # Increased to allow scheduler and early stopping to work
    "PATIENCE": 11,          # Early stopping patience from v3 logic
    "LR_ENCODER": 5e-5,
    "LR_HEAD": 1e-3,
    "SCHEDULER_FACTOR": 0.5, # Reduce LR by half when performance stalls
    "SCHEDULER_PATIENCE": 6, # Wait 2 epochs before reducing LR
    "MAX_BEATS": 6,  
    "NUM_CLASSES": 4,       # N, V, S, F
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_sequential_model():
    """Adjusts the LeadAgnosticTransformer for sequential beat prediction."""
    # Initialize with original PTB-XL class count (5) to load weights
    model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
    
    if os.path.exists(PRETRAINED_WEIGHTS):
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=HP["DEVICE"]))
        print(f"Loaded pre-trained weights from {PRETRAINED_WEIGHTS}")
    
    # Replace head: Output is (MAX_BEATS * NUM_CLASSES)
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
    
    # 2. Optimizer, Loss & Scheduler
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if "classifier" not in n], 'lr': HP["LR_ENCODER"]},
        {'params': model.classifier.parameters(), 'lr': HP["LR_HEAD"]}
    ])
    
    # Monitor Macro F1 to adjust Learning Rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=HP["SCHEDULER_FACTOR"], 
        patience=HP["SCHEDULER_PATIENCE"], verbose=True
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1) # Ignore padding
    
    # 3. Training Loop
    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "f1": [], "lr": []}

    print(f"Starting V1 Training on {HP['DEVICE']}...")

    for epoch in range(HP["EPOCHS"]):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP["DEVICE"]), y.to(HP["DEVICE"])
            optimizer.zero_grad()
            
            # Forward pass and reshape
            logits = model(x).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
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
                        if targets[i, j] != -1: # Filter padding
                            all_preds.append(preds[i, j])
                            all_true.append(targets[i, j])

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)
        
        # Performance Metric for Early Stopping & Scheduler
        report_dict = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        # Step the Scheduler
        scheduler.step(macro_f1)
        current_lr = optimizer.param_groups[1]['lr'] # Monitor Head LR
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["f1"].append(macro_f1)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | F1: {macro_f1:.4f} | LR: {current_lr:.6f}")

        # Early Stopping Logic
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
    print("\n--- GENERATING FINAL REPORTS ---")
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
    
    with open(os.path.join(OUTPUT_DIR, "sequential_report.txt"), "w") as f:
        f.write("LTDB SEQUENTIAL TRANSFER LEARNING (V1 BASELINE + SCHEDULER) REPORT\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    print(report)

    # Plotting results
    plt.figure(figsize=(15, 5))
    
    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.legend()

    # F1 Plot
    plt.subplot(1, 3, 2)
    plt.plot(history["f1"], label='Macro F1', color='green')
    plt.title('Validation Macro F1')
    plt.xlabel('Epoch')
    plt.legend()

    # LR Plot
    plt.subplot(1, 3, 3)
    plt.plot(history["lr"], label='LR (Head)', color='red')
    plt.yscale('log')
    plt.title('Learning Rate Decay')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))

    # Confusion Matrix
    cm = confusion_matrix(final_true, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - V1 with Scheduler')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    print(f"All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_transfer_session()