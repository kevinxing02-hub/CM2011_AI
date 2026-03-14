CM2011_AI
Group work: Filip and Kaiwen

ECG Multi-label Classification & Transfer Learning Pipeline
This repository contains a deep learning pipeline for classifying ECG signals using a Lead-Agnostic Transformer + CNN architecture. The project consists of two phases: pre-training on the PTB-XL dataset and transfer learning on the Long-Term ECG Database (LTDB).

🚀 Execution Order
To run the full experiment, execute the scripts in the following order:

Phase0: 🛠Data Preparation

    Project_Root/
│
├── 🛠raw_data(gitignore)/
│   ├── 🛠ptbxl/                # PTB-XL raw signals and .csv files
│   └── 🛠ltdb/                 # LTDB raw .hea and .dat files
│
├── used_data(gitignore)/
│   ├── data_ptb/            # Processed PTB-XL .pt files (2.5s segments)
│   └── data_ltdb/           # Processed LTDB .pt files (2.5s segments)
│
├── code/
│   ├── PTB_XL_Core_Training/         # Current working directory for scripts
|       ├── 1_data_loader.py   # Step 1: PTB-XL loading
|       ├── 2_split_data.py    # Step 2: 10s -> 2.5s folding
|       ├── 2_train.py         # Step 3: Base training
|       ├── data_utiles.py     # called by train.py
|       └── model.py           # Lead-Agnostic Transformer architecture
│   ├── LTDB_Transfer_Learning/
|       ├── 4_ltdb_loader.py   # Step 4: LTDB loading
|       ├── 5_transfer_train.py# Step 5: Transfer learning
|       └── model.py           # Lead-Agnostic Transformer architecture. COPY FROM PTB_XL
│   
└── results/             # Training outputs
    ├── ptbxl_v1/
    │   └── best_model.pt
    └── ltdb_v1/
        └── best_transfer_model.pt

Phase 1: PTB-XL Core_training
    1_data_loader.py: Loads the raw PTB-XL dataset, aggregates diagnostic labels into superclasses, and saves 10-second signals as .pt tensors.

    2_split_data.py: Segments the 10-second signals into four 2.5-second (250 samples) windows and duplicates labels accordingly.

    3_train.py: Trains the Transformer model on the segmented PTB-XL data.

Phase 2: LTDB Transfer_Learning
    4_ltdb_loader.py:   
                        Processes LTDB records into 2.5-second segments (250 samples) at 100Hz.
                        Uses a sliding window with a 1.75-second stride (0.75-second overlap) for data augmentation.
                        Applies a purity check: Arrhythmias are labeled if they have a 30% majority in the window; "Normal" must be 100% pure.

    5_transfer_train.py:

                        Loads pre-trained weights from Phase 1 and replaces the classification head for LTDB arrhythmia classes.
                        Uses differential learning rates (5e-5 for the backbone, 1e-3 for the head).
                        Generates a classification_report.txt and separate Binary Confusion Matrices for each arrhythmia class to handle multi-label evaluation.

📁 Project Structure
model.py: Defines the LeadAgnosticTransformer which processes each ECG lead independently via CNN before applying attention across leads.

used_data/: (Generated) Contains processed .pt files for both PTB-XL and LTDB.

results/: (Generated) Contains trained model weights (best_transfer_model.pt), loss curves, and per-class metrics.