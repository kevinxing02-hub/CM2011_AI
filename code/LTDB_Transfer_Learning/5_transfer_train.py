"""
PIPELINE STEP 5: Transfer Learning with Metrics
Description:
    Loads pre-trained PTB-XL weights, fine-tunes on LTDB arrhythmia data, 
    and generates performance visualizations and reports.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure the path to model.py is accessible
from model import LeadAgnosticTransformer

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Train_D = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_train.pt'))
Test_D = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_test.pt'))
Read_D = os.path.join(BASE_DIR,'..', '..', 'results', 'ptbxl_v1', 'best_model.pt')
Output_D = os.path.join(BASE_DIR,'..', '..', 'results', 'ltdb_v1')

HP_TRANSFER = {
    "BATCH_SIZE": 32,
    "EPOCHS": 10,  # Adjusted for learning curve visibility
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "PRETRAINED_PATH": Read_D,
    "TRAIN_DATA": Train_D,
    "TEST_DATA": Test_D,
    "OUTPUT_DIR": Output_D
}

def get_transfer_model(num_new_classes):
    """Loads pre-trained model and replaces the classification head."""
    model = LeadAgnosticTransformer(num_classes=5).to(HP_TRANSFER["DEVICE"])
    
    if os.path.exists(HP_TRANSFER["PRETRAINED_PATH"]):
        state_dict = torch.load(HP_TRANSFER["PRETRAINED_PATH"], 
                                map_location=HP_TRANSFER["DEVICE"], 
                                weights_only=False)
        model.load_state_dict(state_dict)
        print("Successfully loaded pre-trained PTB-XL weights.")
    else:
        print("Warning: Pre-trained weights not found. Training from scratch.")

    # Replace head for LTDB specific arrhythmia classes
    model.classifier = nn.Linear(128, num_new_classes)
    return model.to(HP_TRANSFER["DEVICE"])

def get_balanced_loader(data_dict):
    """Handles class imbalance using WeightedRandomSampler."""
    X, y = data_dict['X'], data_dict['y']
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)
    if isinstance(X, np.ndarray): X = torch.from_numpy(X)

    class_indices = torch.argmax(y, dim=1)
    counts = torch.bincount(class_indices)
    weights = 1. / torch.sqrt(counts.float())
    sample_weights = weights[class_indices]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(TensorDataset(X, y), batch_size=HP_TRANSFER["BATCH_SIZE"], sampler=sampler)

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('LTDB Transfer Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "learning_curve.png"))
    plt.close()

def run_transfer_session():
    os.makedirs(HP_TRANSFER["OUTPUT_DIR"], exist_ok=True)
    
    # Load processed LTDB data
    train_dict = torch.load(HP_TRANSFER["TRAIN_DATA"], weights_only=False)
    test_dict = torch.load(HP_TRANSFER["TEST_DATA"], weights_only=False)
    
    train_loader = get_balanced_loader(train_dict)
    
    test_X, test_y = test_dict['X'], test_dict['y']
    if isinstance(test_X, np.ndarray): test_X = torch.from_numpy(test_X)
    if isinstance(test_y, np.ndarray): test_y = torch.from_numpy(test_y)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=HP_TRANSFER["BATCH_SIZE"])
    
    class_names = train_dict['classes']
    model = get_transfer_model(len(class_names))
    
    # Optimizer with Differential Learning Rates
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if "classifier" not in n], 'lr': 5e-5}, 
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    criterion = nn.BCEWithLogitsLoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    print(f"Starting Transfer Learning on {HP_TRANSFER['DEVICE']}...")
    for epoch in range(HP_TRANSFER["EPOCHS"]):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(HP_TRANSFER["DEVICE"]), y.to(HP_TRANSFER["DEVICE"])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                v_loss += criterion(model(x.to(HP_TRANSFER["DEVICE"])), y.to(HP_TRANSFER["DEVICE"])).item()
        
        avg_train = t_loss / len(train_loader)
        avg_val = v_loss / len(test_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}/{HP_TRANSFER['EPOCHS']} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(HP_TRANSFER["OUTPUT_DIR"], "best_transfer_model.pt")
            torch.save(model.state_dict(), save_path)

    # --- FINAL EVALUATION ---
    print("\nGenerating Final Metrics...")
    plot_learning_curves(train_losses, val_losses)
    
    model.load_state_dict(torch.load(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "best_transfer_model.pt")))
    model.eval()
    
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x.to(HP_TRANSFER["DEVICE"]))
            preds = torch.argmax(outputs, dim=1)
            true = torch.argmax(y, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(true.cpu().numpy())

    # Fix class mismatch: use explicit labels based on all classes identified in loader
    label_indices = np.arange(len(class_names))
    
    # Calculate Report
    report = classification_report(
        all_true, 
        all_preds, 
        labels=label_indices, 
        target_names=class_names, 
        zero_division=0
    )
    acc = accuracy_score(all_true, all_preds)

    # Save Report to TXT
    report_path = os.path.join(HP_TRANSFER["OUTPUT_DIR"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("LTDB TRANSFER LEARNING REPORT\n")
        f.write("="*30 + "\n")
        f.write(f"Total Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Print to console
    print("\n--- TEST PERFORMANCE ---")
    print(report)

    # Confusion Matrix Visualization
    cm = confusion_matrix(all_true, all_preds, labels=label_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - LTDB Arrhythmia')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    run_transfer_session()