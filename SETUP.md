# Setup Instructions

This document provides step-by-step instructions for setting up the Oracle Upper Bound method code.

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended)
3. Preprocessed medical imaging data in Blosc2 format

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Core Implementation File

The `oracle_upper_bound_v2.py` file depends on `oracle_upper_bound.py` which contains the core implementation.

**Option A: Copy and Clean (Recommended)**

1. Copy `oracle_upper_bound.py` from the original `inference/` directory to this folder
2. Remove all Chinese comments and strings from the file
3. Update paths in the file:
   - `PREPROCESSED_ROOT`: Path to your preprocessed data
   - `MEDDINO_CHKPT_PATH`: Path to MedDINOv3 checkpoint
   - `DINOV3_HF_MODEL_PATH`: Path to DINOv3 HuggingFace model (if using)

**Option B: Use Provided Script**

A script to automatically clean Chinese from the file can be created if needed.

### 3. Configure Data Paths

Update the following in `oracle_upper_bound.py`:

```python
PREPROCESSED_ROOT = "/path/to/your/preprocessed/data"
MEDDINO_CHKPT_PATH = "/path/to/meddino/model.pth"
DINOV3_HF_MODEL_PATH = "/path/to/dinov3/model"
```

Or set environment variables:

```bash
export PREPROCESSED_ROOT="/path/to/your/preprocessed/data"
export MEDDINO_CHKPT_PATH="/path/to/meddino/model.pth"
```

### 4. Verify Installation

Run the example script to verify everything works:

```bash
python example.py
```

## Data Format Requirements

Your data should be organized as:

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

See `README.md` for detailed data format specifications.

## Model Checkpoints

### MedDINOv3

Download the MedDINOv3 checkpoint and update `MEDDINO_CHKPT_PATH` in `oracle_upper_bound.py`.

### DINOv3 (Optional)

If using DINOv3 HuggingFace model:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-large")
```

Or download locally and set `DINOV3_HF_MODEL_PATH`.

## Troubleshooting

### Import Errors

If you get import errors for `oracle_upper_bound`:
- Ensure `oracle_upper_bound.py` is in the same directory
- Check that all dependencies are installed
- Verify Python path includes this directory

### CUDA Out of Memory

- Reduce `batch_size` in configuration
- Use smaller `K` (fewer slices)
- Use gradient accumulation

### Data Loading Errors

- Verify data paths are correct
- Check that `.b2nd` files are valid Blosc2 format
- Ensure case IDs follow naming convention (ame*/koc*)

## Next Steps

1. Read `README.md` for method overview and workflow
2. Check `example.py` for usage examples
3. Review `CODE_STRUCTURE.md` for code organization

