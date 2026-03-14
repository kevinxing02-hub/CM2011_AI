"""
PIPELINE STEP 5: Refined Transfer Learning
- Fixed: Multi-label sigmoid thresholding
- Fixed: Test set distribution (No sampler on test)
- Added: Separate Binary Confusion Matrices
- Added: Automatic TXT Report Generation
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score

from model import LeadAgnosticTransformer

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Train_D = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_train.pt'))
Test_D = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_test.pt'))
Read_D = os.path.join(BASE_DIR,'..', '..', 'results', 'ptbxl_v1', 'best_model.pt')
Output_D = os.path.join(BASE_DIR,'..', '..', 'results', 'ltdb_v1')

HP_TRANSFER = {
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "PRETRAINED_PATH": Read_D,
    "OUTPUT_DIR": Output_D
}

def get_transfer_model(num_new_classes):
    model = LeadAgnosticTransformer(num_classes=5).to(HP_TRANSFER["DEVICE"])
    if os.path.exists(HP_TRANSFER["PRETRAINED_PATH"]):
        state_dict = torch.load(HP_TRANSFER["PRETRAINED_PATH"], map_location=HP_TRANSFER["DEVICE"], weights_only=False)
        model.load_state_dict(state_dict)
        print("Successfully loaded pre-trained PTB-XL weights.")
    model.classifier = nn.Linear(128, num_new_classes)
    return model.to(HP_TRANSFER["DEVICE"])

def get_balanced_loader(X, y):
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)
    if isinstance(X, np.ndarray): X = torch.from_numpy(X)
    class_indices = torch.argmax(y, dim=1)
    counts = torch.bincount(class_indices)
    weights = 1. / torch.sqrt(counts.float())
    sample_weights = weights[class_indices]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(TensorDataset(X, y), batch_size=HP_TRANSFER["BATCH_SIZE"], sampler=sampler)

def plot_separate_confusion_matrices(y_true, y_pred, class_names):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    cols = 3
    rows = (len(class_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, class_name in enumerate(class_names):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['True Neg', 'True Pos'])
        axes[i].set_title(f'Arrhythmia: {class_name}')
    
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "separate_confusion_matrices.png"))
    plt.close()

def run_transfer_session():
    os.makedirs(HP_TRANSFER["OUTPUT_DIR"], exist_ok=True)
    
    train_dict = torch.load(Train_D, weights_only=False)
    test_dict = torch.load(Test_D, weights_only=False)
    
    # REFINEMENT: Balanced Training vs. Natural Testing
    train_loader = get_balanced_loader(train_dict['X'], train_dict['y'])
    test_loader = DataLoader(TensorDataset(torch.tensor(test_dict['X']).float(), torch.tensor(test_dict['y']).float()), 
                            batch_size=HP_TRANSFER["BATCH_SIZE"])
    
    class_names = train_dict['classes']
    model = get_transfer_model(len(class_names))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(HP_TRANSFER["EPOCHS"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(HP_TRANSFER["DEVICE"]), y.to(HP_TRANSFER["DEVICE"])
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                v_loss += criterion(model(x.to(HP_TRANSFER["DEVICE"])), y.to(HP_TRANSFER["DEVICE"])).item()
        
        avg_val = v_loss/len(test_loader)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val:.4f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(HP_TRANSFER["OUTPUT_DIR"], "best_transfer_model.pt"))

    # --- REFINED EVALUATION ---
    model.load_state_dict(torch.load(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "best_transfer_model.pt")))
    model.eval()
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            # REFINEMENT: Sigmoid thresholding for Multi-label support
            preds = (torch.sigmoid(model(x.to(HP_TRANSFER["DEVICE"]))) > 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)

    # Save TXT Report
    report = classification_report(all_true, all_preds, target_names=class_names, zero_division=0)
    with open(os.path.join(HP_TRANSFER["OUTPUT_DIR"], "final_report.txt"), "w") as f:
        f.write(f"LTDB TRANSFER LEARNING RESULTS\n{'='*30}\n{report}")

    print("\nTraining Complete. Report saved to final_report.txt")
    plot_separate_confusion_matrices(all_true, all_preds, class_names)

if __name__ == "__main__":
    run_transfer_session()