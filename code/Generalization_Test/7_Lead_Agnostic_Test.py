"""
PIPELINE STEP 7: Generalizability Testing (Internal & External)
Description: 
    1. Internal Test: Evaluates lead-agnostic performance on PTB-XL using random lead masking.
    2. External Test: Evaluates clinical robustness on an external dataset (e.g., Chapman-Shaoxing).
"""

import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import your custom architecture
from model import LeadAgnosticTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to your best-performing sequential model (from Step 5)
MODEL_PATH = os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_sequential_v3_v1refined', 'best_model.pt')
# Path to PTB-XL test data for the "Fake" test
PTB_TEST_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb', 'ptbxl_test_250.pt'))
# Path to Chapman or other external data (processed similar to LTDB/PTB)
EXTERNAL_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_external', 'chapman_test.pt'))

OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'generalizability_results')

HP = {
    "BATCH_SIZE": 32,
    "MAX_BEATS": 6,
    "NUM_CLASSES": 4,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class GeneralEval:
    def __init__(self, model_path):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Initialize model with 4 classes as per your sequential transfer logic
        self.model = LeadAgnosticTransformer(num_classes=5).to(HP["DEVICE"])
        # Replace head to match the 4-class sequential output
        self.model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=HP["DEVICE"]))
            print(f"Loaded weights from {model_path}")
        self.model.eval()

    def run_internal_fake_test(self, data_path, num_leads_to_keep=2):
        """
        The 'Fake' Test: Randomly masks leads in PTB-XL to prove lead-agnosticism.
        """
        print(f"\n--- Running Internal 'Fake' Test (Leads kept: {num_leads_to_keep}) ---")
        data = torch.load(data_path, weights_only=False)
        loader = DataLoader(TensorDataset(data['X'], data['y']), batch_size=HP["BATCH_SIZE"])
        
        all_preds, all_true = [], []
        
        with torch.no_grad():
            for x, y in loader:
                # x shape: [Batch, 12, 250]
                batch_size, n_leads, time_steps = x.shape
                
                # Create mask for random lead selection
                mask = torch.zeros_like(x)
                for i in range(batch_size):
                    selected_indices = random.sample(range(n_leads), num_leads_to_keep)
                    mask[i, selected_indices, :] = 1.0
                
                x_masked = (x * mask).to(HP["DEVICE"])
                logits = self.model(x_masked).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                
                # Note: PTB-XL is multi-label, but for this test we treat it 
                # as a 4-class sequence to match your transfer learning head.
                targets = y.numpy()
                for i in range(len(targets)):
                    # For PTB-XL 250ms chunks, we treat the first 'beat' slot as the label
                    # since y was repeated 4 times during segmentation
                    if targets[i].any(): 
                        all_preds.append(preds[i, 0])
                        all_true.append(np.argmax(targets[i]) if targets[i].ndim > 0 else targets[i])

        self.save_results(all_true, all_preds, "internal_fake_test")

    def run_external_real_test(self, data_path):
        """
        The 'Real' Test: Evaluates on a completely new dataset (e.g., Chapman-Shaoxing).
        """
        if not os.path.exists(data_path):
            print(f"External data not found at {data_path}. Skipping.")
            return

        print(f"\n--- Running External 'Real' Test ({os.path.basename(data_path)}) ---")
        data = torch.load(data_path, weights_only=False)
        loader = DataLoader(TensorDataset(data['X'], data['y']), batch_size=HP["BATCH_SIZE"])
        
        all_preds, all_true = [], []
        
        with torch.no_grad():
            for x, y in loader:
                # Feed the new data directly to the model
                logits = self.model(x.to(HP["DEVICE"])).view(-1, HP["MAX_BEATS"], HP["NUM_CLASSES"])
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                targets = y.numpy()
                
                for i in range(len(targets)):
                    for j in range(HP["MAX_BEATS"]):
                        if targets[i, j] != -1: # Ignore padding
                            all_preds.append(preds[i, j])
                            all_true.append(targets[i, j])

        self.save_results(all_true, all_preds, "external_real_test")

    def save_results(self, y_true, y_pred, test_name):
        target_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion/Other']
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        print(report)
        
        # Save Text Report
        with open(os.path.join(OUTPUT_DIR, f"{test_name}_report.txt"), "w") as f:
            f.write(f"GENERALIZABILITY REPORT: {test_name.upper()}\n")
            f.write("="*50 + "\n")
            f.write(report)

        # Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {test_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{test_name}_cm.png"))
        plt.close()

if __name__ == "__main__":
    evaluator = GeneralEval(MODEL_PATH)
    
    # 1. Run Internal "Fake" Test on PTB-XL
    evaluator.run_internal_fake_test(PTB_TEST_PATH, num_leads_to_keep=2)
    
    # 2. Run External "Real" Test (Ensure you have processed Chapman-Shaoxing to .pt first)
    # evaluator.run_external_real_test(EXTERNAL_DATA_PATH)