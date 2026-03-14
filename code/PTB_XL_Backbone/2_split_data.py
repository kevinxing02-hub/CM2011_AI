"""
PIPELINE STEP 2: Data Segmentation (Folding)
Description:
    This script takes the 10-second ECG signals (1000 samples) and splits them 
    into four 2.5-second segments (250 samples each). 
    Labels are duplicated to match the new number of segments.

Order:
    Run this AFTER data_loader.py and BEFORE train.py.
"""

import torch
import os

# ==========================================
# 1. Configuration & Path Setup
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb')
WINDOW_SIZE = 250  # 2.5 seconds at 100Hz !!!!! 

def prepare_folded_ptb():
    train_path = os.path.join(INPUT_DIR, 'ptbxl_train.pt')
    test_path = os.path.join(INPUT_DIR, 'ptbxl_test.pt')

    # Verify files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: Input files not found in {INPUT_DIR}")
        print("Please run data_loader.py first.")
        return

    # Load original 10s data
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    for name, data in [("train", train_data), ("test", test_data)]:
        X, y = data['X'], data['y']
        
        # X shape is [N, Leads, 1000]
        # Split 1000 samples into four 250-sample chunks
        print(f"Processing {name} set (Original shape: {X.shape})...")
        
        chunks = torch.split(X, WINDOW_SIZE, dim=2)
        X_folded = torch.cat(chunks, dim=0)
        
        # Repeat labels 4 times to match the increased sample count [N*4, Classes]
        num_chunks = X.shape[2] // WINDOW_SIZE
        y_folded = torch.cat([y] * num_chunks, dim=0)
        
        # Save as a NEW file to preserve the original 10s data
        save_path = os.path.join(INPUT_DIR, f'ptbxl_{name}_250.pt')
        torch.save({
            'X': X_folded,
            'y': y_folded,
            'classes': data['classes']
        }, save_path)
        
        print(f"-> Folded {name} set saved: {X_folded.shape} at {save_path}")

if __name__ == "__main__":
    prepare_folded_ptb()
    print("\nData segmentation complete. You are ready to run train.py.")