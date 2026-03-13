import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import seaborn as sns

# Import your lead-agnostic architecture
from model import LeadAgnosticTransformer

# ==========================================
# 1. HYPERPARAMETERS & CONFIG
# ==========================================
HP = {
    "LR": 1e-5,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,         
    "D_MODEL": 128,      
    "N_HEADS": 2,
    "N_LAYERS": 2,
    "DROPOUT": 0.2,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "THRESHOLD": 0.5,
    
    # Early Stopping HP
    "PATIENCE": 10,        
    "DELTA": 0.0001,      
    
    #Input DIr
    "train_data_dir" : 'used_data/data_basic_ptb/ptbxl_train_250.pt',
    "test_data_dir" : 'used_data/data_basic_ptb/ptbxl_test_250.pt',

    # Output Management
    "OUTPUT_DIR": "results/basic_ptbxl_experiment_v1",
    "MODEL_NAME": "best_model.pt"
}

# Ensure output directory exists
os.makedirs(HP["OUTPUT_DIR"], exist_ok=True)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def save_loss_plot(train_losses, val_losses, output_path):
    """Generates and saves the Training vs Validation loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()
    print(f"Loss curve saved to {output_path}")

class EarlyStopping:
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==========================================
# 3. TRAINING SESSION FUNCTION
# ==========================================
##### Visual pictures
def plot_multilabel_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """
    Draws and saves visual confusion matrices for multi-label classification.
    y_true, y_pred are (N, num_classes) binary arrays.
    """
    # 1. Calculate the binary matrices for each class
    # cms is shape [num_classes, 2, 2]
    cms = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    
    # 2. Set up the figure (e.g., 1 row of 5 plots)
    fig, axes = plt.subplots(1, num_classes, figsize=(25, 5), sharey=True)
    
    for i, (cm, class_name) in enumerate(zip(cms, class_names)):
        ax = axes[i]
        
        # Format the confusion matrix labels
        # [[TN, FP], [FN, TP]]
        formatted_cm = np.array([
            [cm[0, 0], cm[0, 1]], # Negatives row
            [cm[1, 0], cm[1, 1]]  # Positives row
        ])

        # 3. Create Heatmap using Seaborn
        sns.heatmap(formatted_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    ax=ax, annot_kws={"size": 16}, square=True,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        
        # Annotations
        ax.set_title(f'Class: {class_name}', fontsize=18, fontweight='bold')
        if i == 0:
            ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        
    # Final adjustments and save
    #plt.tight_layout()
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, 'multilabel_confusion_matrices.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visual Confusion Matrices saved to {save_path}")


def run_training_session(model, train_loader, test_loader, hp, class_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["LR"])
    criterion = nn.BCEWithLogitsLoss()
    
    model_save_path = os.path.join(hp["OUTPUT_DIR"], hp["MODEL_NAME"])
    early_stopping = EarlyStopping(patience=hp["PATIENCE"], delta=hp["DELTA"], path=model_save_path)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(hp["EPOCHS"]):
        model.train()
        train_loss = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(hp["DEVICE"]), labels.to(hp["DEVICE"])
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(hp["DEVICE"]), labels.to(hp["DEVICE"])
                outputs = model(signals)
                val_loss += criterion(outputs, labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        
        print(f"Epoch [{epoch+1}/{hp['EPOCHS']}] | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            break

    # Save visualization
    save_loss_plot(history["train_loss"], history["val_loss"], hp["OUTPUT_DIR"])

    # Final Evaluation with Best Weights
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(hp["DEVICE"]), labels.to(hp["DEVICE"])
            outputs = model(signals)
            preds = torch.sigmoid(outputs) > hp["THRESHOLD"]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # plot pictures confusion matrix in different labels
    plot_multilabel_confusion_matrices(all_labels, all_preds, class_names, hp["OUTPUT_DIR"])

    # Save text results to file
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(hp["OUTPUT_DIR"], "classification_report.txt"), "w") as f:
        f.write(report)
    
    print("\nTraining Complete. Results saved in:", hp["OUTPUT_DIR"])
    print(report)
# ==========================================
# 4. DATA LOADING & EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Actually LOAD the .pt files into memory
    # HP['train_data_dir'] is just a string; torch.load() turns it into a dictionary
    train_dict = torch.load(HP['train_data_dir'], weights_only=False)
    test_dict = torch.load(HP['test_data_dir'], weights_only=False)
    
    # 2. Create Loaders using the dictionary keys 'X' and 'y'
    train_loader = DataLoader(
        TensorDataset(train_dict['X'], train_dict['y']), 
        batch_size=HP["BATCH_SIZE"], 
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_dict['X'], test_dict['y']), 
        batch_size=HP["BATCH_SIZE"]
    )
    
    # 3. Initialize Model using the 'classes' key from the loaded dictionary
    num_classes = len(train_dict['classes'])
    class_names = train_dict['classes']
    
    model = LeadAgnosticTransformer(
        num_classes=num_classes, 
        d_model=HP["D_MODEL"], 
        nhead=HP["N_HEADS"], 
        num_layers=HP["N_LAYERS"]
    ).to(HP["DEVICE"])

    # 4. Start Session
    run_training_session(model, train_loader, test_loader, HP, class_names)