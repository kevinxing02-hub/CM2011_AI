import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import sys
sys.path.append(r"./code_basic_ptb")
from model import LeadAgnosticTransformer

from torch.utils.data import WeightedRandomSampler


# ==========================================
# 1. EARLY STOPPING CLASS
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model.pt'):
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    cms = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    
    if num_classes == 1: axes = [axes] # Handle single class case
    
    for i, (cm, class_name) in enumerate(zip(cms, class_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Class: {class_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

# ==========================================
# 3. TRAINING SESSION
# ==========================================
HP_TRANSFER = {
    "LR": 1e-4,
    "BATCH_SIZE": 32,
    "EPOCHS": 40,
    "PATIENCE": 5,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "PRETRAINED_PATH": "results/basic_ptbxl_experiment_v1/best_model.pt",
    "TRAIN_DATA": "used_data/data_transfer_ltdb/ltdb_train.pt",
    "TEST_DATA": "used_data/data_transfer_ltdb/ltdb_test.pt",
    "OUTPUT_DIR": "results/transfer_ltdb_v1",
    "D_MODEL": 128, "N_HEADS": 2, "N_LAYERS": 2
}
'''def get_transfer_model(checkpoint_path, num_new_classes, hp):
    model = LeadAgnosticTransformer(num_classes=5, d_model=hp["D_MODEL"], nhead=hp["N_HEADS"], num_layers=hp["N_LAYERS"])
    state_dict = torch.load(checkpoint_path, map_location=hp["DEVICE"], weights_only=False)
    model.load_state_dict(state_dict)
    for param in model.parameters(): param.requires_grad = False
    model.classifier = nn.Linear(hp["D_MODEL"], num_new_classes)
    return model.to(hp["DEVICE"])'''


def get_transfer_model(checkpoint_path, num_new_classes, hp):
    model = LeadAgnosticTransformer(
        num_classes=5, 
        d_model=hp["D_MODEL"],
        nhead=hp["N_HEADS"],
        num_layers=hp["N_LAYERS"]
    )
    
    state_dict = torch.load(checkpoint_path, map_location=hp["DEVICE"], weights_only=False)
    model.load_state_dict(state_dict)
    
    # --- UNFREEZE STRATEGY ---
    # 1. First, set everything to trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # 2. (Optional) If you want to keep the first CNN layer frozen 
    # (because basic edge/curve detection is universal):
    # for param in model.cnn[0].parameters():
    #     param.requires_grad = False

    # 3. Replace the Head (this is already trainable by default)
    model.classifier = nn.Linear(hp["D_MODEL"], num_new_classes)
    
    return model.to(hp["DEVICE"])

def get_balanced_loader(X, y, batch_size):
    # 1. Calculate weights per sample
    # y is shape [N, num_classes]
    class_indices = torch.argmax(y, dim=1)
    counts = torch.bincount(class_indices)
    
    # Weight = 1 / frequency
    weights = 1. / torch.sqrt(counts.float())
    sample_weights = weights[class_indices]
    
    # 2. Create the Sampler
    # 'replacement=True' allows the rare 'S' and 'J' to be picked multiple times in one epoch
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def run_transfer_session():
    os.makedirs(HP_TRANSFER["OUTPUT_DIR"], exist_ok=True)
    train_dict = torch.load(HP_TRANSFER["TRAIN_DATA"], weights_only=False)
    test_dict = torch.load(HP_TRANSFER["TEST_DATA"], weights_only=False)
    
    
    # Usage in run_transfer_session:
    train_loader = get_balanced_loader(train_dict['X'], train_dict['y'], HP_TRANSFER["BATCH_SIZE"])

    #train_loader = DataLoader(TensorDataset(train_dict['X'], train_dict['y']), batch_size=HP_TRANSFER["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_dict['X'], test_dict['y']), batch_size=HP_TRANSFER["BATCH_SIZE"])
    
    class_names = train_dict['classes']
    model = get_transfer_model(HP_TRANSFER["PRETRAINED_PATH"], len(class_names), HP_TRANSFER)
    
    # Inside run_transfer_session():

    # Separate the parameters
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = model.classifier.parameters()

    # Pass them to the optimizer with different rates
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 5e-5}, # Very slow for the CNN/Transformer
        {'params': head_params, 'lr': 1e-3}      # Normal speed for the new Head
    ])

    '''optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=HP_TRANSFER["LR"])'''
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize Early Stopping
    checkpoint_path = os.path.join(HP_TRANSFER["OUTPUT_DIR"], "best_transfer_model.pt")
    early_stopping = EarlyStopping(patience=HP_TRANSFER["PATIENCE"], path=checkpoint_path)

    for epoch in range(HP_TRANSFER["EPOCHS"]):
        # --- TRAINING PASS ---
        model.train()
        train_loss = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(HP_TRANSFER["DEVICE"]), labels.to(HP_TRANSFER["DEVICE"])
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- VALIDATION PASS (This gives you the second loss number) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(HP_TRANSFER["DEVICE"]), labels.to(HP_TRANSFER["DEVICE"])
                val_loss += criterion(model(signals), labels).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{HP_TRANSFER['EPOCHS']}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Check Early Stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # --- FINAL EVALUATION & CONFUSION MATRIX ---
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(HP_TRANSFER["DEVICE"])
            outputs = torch.sigmoid(model(signals)) > 0.5
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    y_true, y_pred = torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    
    print("\nTransfer Learning Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    plot_confusion_matrices(y_true, y_pred, class_names, HP_TRANSFER["OUTPUT_DIR"])
    print(f"Confusion matrices saved to {HP_TRANSFER['OUTPUT_DIR']}")

if __name__ == "__main__":
    run_transfer_session()