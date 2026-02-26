# Oracle library for MICCAI baselines/ablations
from .evidence_readout import (
    compute_patch_mask_weights,
    extract_slice_tokens,
    maskpool_embeddings,
    MASKPOOL_VARIANTS,
)

__all__ = [
    "compute_patch_mask_weights",
    "extract_slice_tokens", 
    "maskpool_embeddings",
    "MASKPOOL_VARIANTS",
]

