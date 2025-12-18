# Multimodal Embedding-Based CTR Prediction System

A high-performance Click-Through Rate (CTR) prediction system that leverages **multimodal embeddings** by combining textual and visual information from product data. The system integrates state-of-the-art representation learning with a deep neural CTR architecture to achieve **0.9251 validation AUC**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Citation](#citation)

---

## 🎯 Overview

This project processes **91,718 items** with textual descriptions and product images to predict user engagement. The solution consists of two main components:

1. **Multimodal Embedding Generator**: Fuses text and image embeddings into compact 128-dimensional representations
2. **Deep Learning CTR Model**: Predicts click-through rates using attention mechanisms and feature interactions

### Performance Metrics

| Metric             | Score              |
|--------------------|--------------------|
| **Validation AUC** | **0.9251**         |
| Training AUC       | 0.9999             |
| Model Parameters   | ~70.8M             |
| Training Epochs    | 12 (early stopped) |

---

## ✨ Key Features

- 🖼️ **Multimodal Fusion**: Combines CLIP visual features with BERT text embeddings
- 🎯 **Attention Pooling**: Adaptive weighting of user interaction sequences
- 🔗 **Feature Interactions**: Bilinear layers for user-item and sequence-item relationships
- 📊 **Advanced Regularization**: Label smoothing, dropout, and early stopping
- ⚡ **Differential Learning Rates**: Separate rates for pre-trained and trainable components

---

## 🏗️ Architecture

### Component 1: Multimodal Embedding Generator

```
Input Data
├── Text: Item titles (91,717 items)
└── Images: Product photos (image{item_id}.jpg)
         ↓
    Encoding Models
    ├── Sentence-BERT (all-MiniLM-L6-v2) → 384-d
    └── CLIP ResNet-50 → 1,024-d
         ↓
    Concatenation → 1,408-d
         ↓
    PCA Compression → 128-d
         ↓
    Output: item_info_fused_multimodal.parquet
```

**Key Techniques:**
- **Text Embeddings**: Sentence-BERT captures semantic meaning from item titles
- **Image Embeddings**: CLIP extracts high-level visual features
- **Dimensionality Reduction**: PCA preserves ~90% variance while reducing to 128-d
- **Normalization**: Standardized embeddings for stable training

### Component 2: Deep CTR Prediction Model

```
ImprovedMMCTRModel Architecture
├── Embedding Layers
│   ├── User Embeddings (64-d)
│   ├── Item Embeddings (64-d)
│   ├── Categorical Features (32-d each)
│   └── Multimodal Transform (128-d → 64-d)
├── Attention Mechanism
│   └── AttentionPooling for sequences
├── Feature Interactions
│   ├── User-Item Bilinear (64-d)
│   └── Sequence-Item Bilinear (64-d)
└── Deep MLP
    └── 448 → 512 → 256 → 128 → 1
```

**Model Components:**
- **Embeddings**: User (64-d), Item (64-d), Likes (32-d), Views (32-d)
- **Attention**: Custom pooling layer for variable-length sequences
- **Interactions**: Bilinear layers capture cross-feature relationships
- **MLP**: 4-layer network with batch normalization and dropout

---

## 📊 Dataset

### Data Splits

| Split      | Samples   | Description                         |
|------------|-----------|-------------------------------------|
| Training   | 3,600,000 | User-item interactions for training |
| Validation | 10,000    | Held-out set for model selection    |
| Test       | 379,142   | Final evaluation set                |

### Data Structure

**Input Files:**
- `item_info.parquet` (91,718 rows): Base item information
- `item_feature.parquet` (91,717 rows): Item titles and features
- `item_images/`: Product photos named as `image{item_id}.jpg`
- `train.parquet`, `valid.parquet`, `test.parquet`: Interaction data

**Features:**
- `user_id`: User identifier (1M unique users)
- `item_id`: Item identifier (91,718 items)
- `item_seq`: User's historical item interactions
- `likes_level`: Categorical engagement level (0-10)
- `views_level`: Categorical view level (0-10)
- `label`: Binary target (click/no-click)

---

## 📈 Results

### Model Performance

```
Validation AUC: 0.9251
Training AUC: 0.9999
Early Stopping: Epoch 12/30
```

### Test Predictions Statistics

| Metric  | Value |
|---------|-------|
| Minimum | 0.014 |
| Maximum | 0.981 |
| Mean    | 0.574 |

### Training Curve

```
Epoch 01: Val AUC 0.8868 ✅
Epoch 02: Val AUC 0.9003 ✅
Epoch 04: Val AUC 0.9188 ✅
Epoch 05: Val AUC 0.9251 ✅ (Best)
Epoch 12: Early stopping triggered
```

---

## 🛠️ Installation

### Requirements

```bash
Python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### Install Dependencies

```bash
# Install PyTorch (visit pytorch.org for your CUDA version)
pip install torch torchvision

# Install other dependencies
pip install pandas numpy scikit-learn pillow tqdm

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install Sentence Transformers
pip install sentence-transformers

# Downgrade protobuf if needed
pip install protobuf==3.20.3

The usage of a GPU is crucial 
```


## 🚀 Usage

### Step 1: Generate Multimodal Embeddings

```python
# Run: embedding_generation(1).ipynb

# This script will:
# 1. Load item titles and images
# 2. Generate text embeddings using Sentence-BERT
# 3. Generate image embeddings using CLIP
# 4. Fuse and compress to 128 dimensions using PCA
# 5. Save to item_info_fused_multimodal.parquet

# Expected runtime: ~30-45 minutes on GPU
```

**Output:** `item_info_fused_multimodal.parquet`

### Step 2: Train CTR Model

```python
# Run: final-ctr-improved.ipynb

# This script will:
# 1. Load multimodal embeddings
# 2. Prepare train/validation/test datasets
# 3. Train improved MMCTR Model with early stopping
# 4. Generate predictions on the test set
# 5. Save best_model.pt and submission.csv

# Expected runtime: ~2-3 hours on GPU
```

**Outputs:** 
- `best_model.pt` (trained model)
- `submission.csv` (test predictions)

### Step 3: Generate Submission

The model automatically creates `submission.csv` with the following format:

```csv
ID,Task1&2
0,0.573421
1,0.824156
...
```

---

## 🔬 Model Details

### Training Strategy

**Loss Function:**
- Binary Cross-Entropy with Label Smoothing (smoothing=0.05)
- Prevents overconfidence and improves generalization

**Optimizer:**
- AdamW with differential learning rates
  - Main parameters: `lr=2e-3`, `weight_decay=1e-4`
  - Multimodal transform: `lr=5e-4`, `weight_decay=1e-5`

**Learning Rate Schedule:**
- Cosine Annealing with Warm Restarts
  - `T_0=5`, `T_mult=2`, `eta_min=1e-6`

**Regularization:**
- Dropout: 0.3
- Gradient clipping: max_norm=1.0
- Early stopping: patience=7 epochs
- Weight decay: 1e-4

### Hyperparameters

```python
BATCH_SIZE = 2048
EMBEDDING_DIM = 64
EPOCHS = 30
PATIENCE = 7
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
```

---

## 🧪 Key Innovations

1. **Multimodal Representation Learning**
   - Leverages both textual and visual product information
   - Compact 128-d embeddings capture rich semantic features

2. **Attention-Based Sequence Modeling**
   - Dynamic weighting of user interaction history
   - Properly handles variable-length sequences with masking

3. **Bilinear Feature Interactions**
   - Captures complex non-linear relationships
   - User-item and sequence-item cross-features

4. **Safe Fine-Tuning Strategy**
   - Differential learning rates preserve pre-trained knowledge
   - Lower LR for multimodal transform prevents catastrophic forgetting

5. **Robust Regularization Pipeline**
   - Multiple regularization techniques prevent overfitting
   - Achieves 0.9251 val AUC despite 70M+ parameters


---

## 🎓 Technical Details

### Embedding Dimensions

| Component            | Input         | Output  |
|----------------------|---------------|---------|
| Text (Sentence-BERT) | Item title    | 384-d   |
| Image (CLIP)         | Product photo | 1,024-d |
| Concatenation        | -             | 1,408-d |
| PCA                  | 1,408-d       | 128-d   |
| Transform            | 128-d         | 64-d    |

### Model Capacity

```
Total Parameters: 70,802,562
├── User Embeddings: 64,000,064
├── Item Embeddings: 5,869,952
├── Feature Embeddings: 704
├── Attention Layers: 12,416
├── Bilinear Layers: 16,512
└── MLP Layers: 903,937
```

---

## 📦 Dataset

The **Microlens dataset** is publicly available on Kaggle. It contains all the required files for this project, including item information, features, images, and train/validation/test splits.

**Dataset Link:** [https://www.kaggle.com/datasets/hajarelmahjouby/microlens/data](https://www.kaggle.com/datasets/hajarelmahjouby/microlens/data)
To use the dataset:
1. Visit the link above
2. Click "Download" or add it as a data source in your Kaggle notebook
3. Follow the usage instructions in this README



---

**⭐ Thank you**
