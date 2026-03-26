"""
PIPELINE STEP 7: Lead-Agnostic Generalization Test
Description:
    Tests the model's consistency across different lead combinations.
    If the model predicts the same class regardless of which leads are used,
    it proves the architecture is truly lead-agnostic.
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
# Points to your best model from the Focal Loss experiment
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_final', 'best_sequential_model.pt'))
# Points to PTB-XL test data (12-lead source)
PTB_TEST_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb', 'ptbxl_test_250.pt'))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'generalization_test')

HP = {
    "BATCH_SIZE": 32,
    "MAX_BEATS": 6,
    "NUM_CLASSES": 4,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class GeneralEval:
    def __init__(self, model_path):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 1. Initialize Model
        self.model = LeadAgnosticTransformer(num_classes=5)
        # 2. Reconstruct the Sequential Head (4 classes, 6 beats) to match your Step 5 script
        self.model.classifier = nn.Linear(128, HP["MAX_BEATS"] * HP["NUM_CLASSES"])
        
        # 3. Move to Device BEFORE loading weights
        self.model.to(HP["DEVICE"])
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=HP["DEVICE"]))
            print(f"✅ Successfully loaded weights from: {model_path}")
        else:
            raise FileNotFoundError(f"❌ Model weights not found at {model_path}")
            
        self.model.eval()

    def run_consistency_test(self, data_path):
        """
        Calculates the 'Lead-Agnostic Consistency Score'.
        Compares predictions using 'Standard Leads' (II, V1) vs 'Random Leads'.
        """
        print(f"\n--- 🧪 Running Lead-Agnostic Consistency Test ---")
        if not os.path.exists(data_path):
            print(f"❌ Data not found at {data_path}")
            return

        data = torch.load(data_path, weights_only=False)
        loader = DataLoader(TensorDataset(data['X'], data['y']), batch_size=HP["BATCH_SIZE"])
        
        matches = 0
        total_beats = 0
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(HP["DEVICE"]) # [Batch, 12, 250]
                
                # TEST A: Standard leads (Lead 1 and Lead 7 usually correspond to II and V1 in PTB-XL)
                x_std = torch.zeros_like(x)
                x_std[:, [1, 7], :] = x[:, [1, 7], :]
                logits_std = self.model(x_std).view(-1, HP["NUM_CLASSES"])
                pred_std = torch.argmax(logits_std, dim=1)
                
                # TEST B: Randomly selected leads for every batch
                x_rnd = torch.zeros_like(x)
                # Pick 2 different random leads from the 12 available
                idx = random.sample(range(12), 2)
                x_rnd[:, idx, :] = x[:, idx, :]
                logits_rnd = self.model(x_rnd).view(-1, HP["NUM_CLASSES"])
                pred_rnd = torch.argmax(logits_rnd, dim=1)
                
                # Compare if the model gives the same classification
                matches += (pred_std == pred_rnd).sum().item()
                total_beats += pred_std.size(0)

        consistency = (matches / total_beats) * 100
        print(f"📊 Results:")
        print(f"   Total Beats Analyzed: {total_beats}")
        print(f"   Consistency Score: {consistency:.2f}%")
        print(f"\nConclusion: The model is {consistency:.2f}% lead-invariant.")
        
        # Save results to a text file
        with open(os.path.join(OUTPUT_DIR, "consistency_test_results.txt"), "w") as f:
            f.write("LEAD-AGNOSTIC CONSISTENCY TEST\n")
            f.write("==============================\n")
            f.write(f"Consistency Score: {consistency:.2f}%\n")
            f.write(f"Total Samples: {total_beats}\n")
            f.write("The score measures how often the model makes the same prediction\n")
            f.write("using different lead combinations (Standard vs Random).")

if __name__ == "__main__":
    evaluator = GeneralEval(MODEL_PATH)
    
    # Run the test (No arguments needed now, avoids the TypeError)
    evaluator.run_consistency_test(PTB_TEST_PATH)