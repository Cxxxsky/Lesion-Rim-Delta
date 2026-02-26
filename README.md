# MedDINOv3 Inference: Oracle Upper Bound Method

This repository provides the inference code for the Oracle Upper Bound method used in medical image classification tasks. The method uses ground-truth segmentation masks for slice selection to estimate the theoretical upper bound of selection strategies.

## Overview

The Oracle Upper Bound method is designed for binary classification tasks (e.g., AME vs OKC) using 3D medical imaging data. The workflow consists of:

1. **Slice Selection**: Compute lesion evidence scores for all slices based on GT mask area, then select top-K slices
2. **Feature Extraction**: Extract features from selected slices using Vision Transformer (ViT) backbones
3. **Feature Aggregation**: Aggregate slice-level features into case-level embeddings
4. **Classification**: Train a lightweight classifier (MLP/MIL) for final classification

## Method Workflow

### Step 1: Data Preparation

The method expects data in the following format:

```
preprocessed_root/
└── nnUNetPlans_2d/
    ├── {case_id}.b2nd          # 3D volume (C, Z, Y, X) or (Z, Y, X)
    ├── {case_id}_seg.b2nd      # Segmentation mask (same shape)
    └── {case_id}.pkl           # Metadata (optional)
```

**Data Format Details:**
- **Image files** (`{case_id}.b2nd`): Blosc2-compressed numpy arrays
  - Shape: `(C, Z, Y, X)` for multi-channel or `(Z, Y, X)` for single-channel
  - Data type: float32
  - Case ID naming: `ame001`, `ame002`, ... (label=1) or `koc001`, `koc002`, ... (label=0)

- **Segmentation masks** (`{case_id}_seg.b2nd`): Binary masks
  - Shape: Same as image file
  - Values: 0 (background), 1 (lesion), 2 (tooth, optional)
  - Used for oracle slice selection (GT mask area)

### Step 2: Slice Selection

For each case, compute oracle scores for all slices:

```python
scores = compute_oracle_slice_scores(seg_path)  # (Z,) array of mask areas
selected_indices = select_slices(scores, strategy="topk", K=3)
```

**Selection Strategies:**
- `"topk"`: Select K slices with highest scores (allows duplicates)
- `"topk_unique"`: Select K unique slices with highest scores
- `"window"`: Select continuous window around argmax

### Step 3: Feature Extraction

Extract features from selected slices using one of three modes:

#### Mode 1: `full_global` (Baseline)
- Input: Full slice image
- Processing: Resize with aspect ratio preservation + padding
- Output: Global average pooling of patch tokens → (D,)

#### Mode 2: `full_maskpool` (Main Method)
- Input: Full slice image + GT mask
- Processing: 
  1. Extract patch tokens from full image
  2. Compute patch-level weights for lesion and ring regions
  3. Apply mask-weighted pooling
- Output: Variant-dependent embedding
  - `"lesion"`: (D,) - lesion pooled features
  - `"delta"`: (D,) - lesion - ring (context-aware)
  - `"lesion+global"`: (2D,) - concat(lesion, global)
  - `"lesion+global+delta"`: (3D,) - concat(lesion, global, delta)

#### Mode 3: `crop_roi` (Ablation)
- Input: Cropped ROI based on mask bounding box
- Processing: Crop → resize → global pooling
- Output: (D,) or tiled to match variant dimension

**Mask Pooling Details:**
- **Lesion region**: Pixels with mask > 0
- **Ring region**: Dilated mask - original mask (context around lesion)
- **Ring width**: Configurable dilation radius (default: 20 pixels)

### Step 4: Feature Aggregation

Aggregate K slice embeddings into case-level embedding:

```python
# Mean aggregation
case_embedding = slice_embeddings.mean(axis=0)  # (D,)

# Weighted aggregation (by oracle score)
weights = slice_scores / slice_scores.sum()
case_embedding = (slice_embeddings * weights[:, np.newaxis]).sum(axis=0)

# Max aggregation
case_embedding = slice_embeddings.max(axis=0)
```

### Step 5: Classification

Train a classifier on case-level embeddings:

**Non-MIL Classifiers** (for aggregated embeddings):
- `"mlp"`: Multi-layer perceptron with dropout
- `"linear"`: Simple linear classifier
- `"learnable_delta"`: Learnable alpha for delta computation

**MIL Classifiers** (for per-slice embeddings):
- `"attention_mil"`: Attention-based MIL
- `"gated_attention_mil"`: Gated attention MIL
- `"transmil"`: Transformer-based MIL

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd meddinov3_inference

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- scikit-learn
- matplotlib
- blosc2
- transformers (for DINOv3 HuggingFace support)

See `requirements.txt` for complete list.

## Usage

### Basic Usage

```python
from oracle_upper_bound_v2 import run_experiment, DEFAULT_CONFIG

# Run with default configuration
results = run_experiment()

# Custom configuration
results = run_experiment(
    backbone="meddino",           # or "dinov3_hf"
    mode="full_maskpool",
    maskpool_variant="delta",
    K=3,
    aggregation="mean",
    classifier_type="mlp",
    dropout=0.75,
    out_root="./outputs",
)
```

### Configuration Parameters

**Backbone:**
- `"meddino"`: MedDINOv3 ViT-B-16 (768-dim, CT pretrained, input=2048)
- `"dinov3_hf"`: DINOv3 ViT-H+-16 (1280-dim, natural image pretrained, input=518)

**Feature Extraction:**
- `mode`: `"full_global"`, `"full_maskpool"`, or `"crop_roi"`
- `maskpool_variant`: `"lesion"`, `"delta"`, `"lesion+global"`, `"lesion+global+delta"`, `"lesion+ring"`
- `rim_width`: Ring dilation width in pixels (default: 20)

**Slice Selection:**
- `selection_strategy`: `"topk"`, `"topk_unique"`, or `"window"`
- `K`: Number of slices to select (default: 3)
- `aggregation`: `"mean"`, `"weighted"`, or `"max"`

**Classifier:**
- `classifier_type`: `"mlp"`, `"linear"`, `"learnable_delta"`, `"attention_mil"`, `"gated_attention_mil"`, `"transmil"`
- `hidden_dims`: List of hidden layer dimensions (default: [128, 64])
- `dropout`: Dropout rate (default: 0.75)

**Training:**
- `epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay (default: 1e-2)
- `patience`: Early stopping patience (default: 30)
- `n_folds`: Number of CV folds (default: 5)

## Data Format Specification

### Input Data Structure

```
preprocessed_root/
└── nnUNetPlans_2d/
    ├── ame001.b2nd
    ├── ame001_seg.b2nd
    ├── ame002.b2nd
    ├── ame002_seg.b2nd
    ├── koc001.b2nd
    ├── koc001_seg.b2nd
    └── ...
```

### File Format: Blosc2 (.b2nd)

- **Compression**: Blosc2 format (fast compression for scientific data)
- **Shape**: 
  - Multi-channel: `(C, Z, Y, X)` where C=channels, Z=slices, Y=height, X=width
  - Single-channel: `(Z, Y, X)`
- **Data type**: `float32` for images, `int32` or `uint8` for masks
- **Loading**: Use `blosc2.open(path, mode="r")` to read

### Segmentation Mask Format

- **Values**:
  - `0`: Background
  - `1`: Lesion (target region)
  - `2`: Tooth (optional, ignored in current implementation)
- **Shape**: Must match image file shape
- **Usage**: Used for oracle slice selection (computing lesion area per slice)

### Case ID Convention

- **AME cases** (label=1): `ame001`, `ame002`, `ame003`, ...
- **KOC cases** (label=0): `koc001`, `koc002`, `koc003`, ...

The method automatically discovers cases by scanning the directory and inferring labels from case ID prefixes.

## Output Format

The method outputs:

1. **Cross-validation results** (`oracle_cv_results.json`):
```json
{
    "mean_accuracy": 0.8256,
    "std_accuracy": 0.0234,
    "mean_auc": 0.8886,
    "std_auc": 0.0156,
    "overall_accuracy": 0.8300,
    "overall_auc": 0.8900,
    "overfit_gap": 0.0123,
    "fold_results": [...]
}
```

2. **Training curves** (`cv_training_curves.png`): Visualization of training/validation metrics across folds

3. **Feature cache** (`feature_cache/features_*.pkl`): Cached features for faster re-runs

## Feature Caching

Features are automatically cached based on a hash of:
- Case list
- Selection strategy and K
- Mode and variant
- Input size, ROI margin, ring width
- Backbone type
- Mask source (oracle vs predicted)

To clear cache and re-extract features:
```bash
rm -rf feature_cache/
```

## Examples

See `example.py` for a complete working example.



## License

[Specify your license here]

## Contact

[Your contact information]

