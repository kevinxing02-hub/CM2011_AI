# CM2011_AI  
**Group Work:** Filip and Kaiwen  

## ECG Multi-label Classification & Transfer Learning Pipeline

This repository contains a **deep learning pipeline for classifying ECG signals** using a **Lead-Agnostic Transformer + CNN architecture**.

The project is divided into two major phases:

1. **Pre-training on the PTB-XL dataset**
2. **Transfer learning on the Long-Term ECG Database (LTDB)**

---

# 🚀 Execution Order

To reproduce the full experiment, run the scripts **in the following order**.

---

# Phase 0: 🛠 Data Preparation

⭐ **Only prepare items marked with a star**
```
Project_Root/
│
├── raw_data⭐/ (gitignored)
│   ├── ptbxl⭐/                # Paste the PTB-XL raw dataset here and rename the folder to "ptbxl"
│   └── ltdb⭐/                 # Paste the LTDB-MIT raw dataset here and rename the folder to "ltdb"
│
├── used_data/ (gitignored)
│   ├── data_ptb/             # Processed PTB-XL .pt files (2.5s segments)
│   └── data_ltdb/            # Processed LTDB .pt files (2.5s segments)
│
├── code/
│   ├── PTB_XL_Core_Training/
│   │   ├── 1_data_loader.py   # Step 1: PTB-XL loading
│   │   ├── 2_split_data.py    # Step 2: 10s → 2.5s segmentation
│   │   ├── 3_train.py         # Step 3: Base model training
│   │   ├── data_utiles.py     # Utility functions used during training
│   │   └── model.py           # Lead-Agnostic Transformer architecture
│   │
│   ├── LTDB_Transfer_Learning/
│   │   ├── 4_ltdb_loader.py    # Step 4: LTDB loading
│   │   ├── 5_transfer_train.py # Step 5: Transfer learning
│   │   └── model.py            # Copy of PTB-XL model architecture
│
└── results/
    ├── ptbxl_v1/
    │   └── best_model.pt
    └── ltdb_v1/
        └── best_transfer_model.pt
```

---

# Phase 1: PTB-XL Core Training

### `1_data_loader.py`
- Loads the raw **PTB-XL dataset**
- Aggregates diagnostic labels into **superclasses**
- Saves each **10-second ECG signal** as `.pt` tensors

### `2_split_data.py`
- Segments each **10-second signal** into **four 2.5-second windows**
- Each window contains **250 samples**
- Corresponding labels are duplicated for each segment

### `3_train.py`
- Trains the **Lead-Agnostic Transformer model**
- Uses segmented PTB-XL data as input

---

# Phase 2: LTDB Transfer Learning

### `4_ltdb_loader.py`

Processes LTDB ECG records into training segments.

Steps:
- Converts records into **2.5-second segments (250 samples) at 100 Hz**
- Applies **sliding window segmentation**
  - Window length: **2.5 seconds**
  - Stride: **1.75 seconds**
  - Overlap: **0.75 seconds**
- Performs **purity checks for labels**
  - Arrhythmia classes require **≥30% presence** within the window
  - **Normal rhythm must be 100% pure**

---

### `5_transfer_train.py`

Performs transfer learning from the PTB-XL model.

Key features:

- Loads **pre-trained weights** from Phase 1
- Replaces the **classification head** for LTDB arrhythmia classes
- Uses **differential learning rates**
  - Backbone: `5e-5`
  - Classification head: `1e-3`

Evaluation outputs:
- `classification_report.txt`
- **Binary Confusion Matrices per arrhythmia class**

This approach handles the **multi-label nature** of ECG classification.

---

# 📌 Notes

- Raw datasets **PTB-XL** and **LTDB** are **not included in the repository** due to size constraints.
- Place them inside the `raw_data/` directory before running the pipeline.

---

# 👥 Authors

**Filip**  
**Kaiwen**

Course Project – **CM2011_AI**