"""
================================================================================
Evidence Readout Module for MICCAI Baselines/Ablations
================================================================================

Provides dimension-stable feature extraction with configurable pooling variants.

Functions:
    - compute_patch_mask_weights: Compute patch-level lesion and ring weights
    - extract_slice_tokens: Full-slice forward pass, returns patch tokens
    - maskpool_embeddings: Dimension-stable masked pooling with variants

Author: YYX Research
================================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Literal, Optional

# Supported maskpool variants and their output dimensions (as multiplier of base D)
MASKPOOL_VARIANTS = {
    "lesion": 1,           # D
    "delta": 1,            # D (lesion - ring)
    "delta_global": 1,     # D (lesion - global) - for ablation: ring vs global context
    "lesion+global": 2,    # 2D (lesion, global)
    "lesion+ring": 2,      # 2D (lesion, ring) - for learnable delta
    "lesion+global+delta": 3,  # 3D (lesion, global, delta)
}


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Simple dilation using iterative max pooling (avoids large kernel slowness).
    
    Args:
        mask: Binary mask (H, W)
        pixels: Dilation radius in pixels
        
    Returns:
        Dilated binary mask
    """
    if pixels <= 0:
        return mask.copy()
    
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    result = mask_tensor
    remaining = pixels
    while remaining > 0:
        k = min(remaining * 2 + 1, 15)  # max kernel size 15
        p = k // 2
        result = F.max_pool2d(result, k, stride=1, padding=p)
        remaining -= p
    
    return result.squeeze().numpy() > 0


def compute_patch_mask_weights(
    mask_yx: np.ndarray,
    input_size: int,
    patch_size: int,
    resize_info: Tuple[float, int, int, int, int],
    ring_dilate_pixels: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-patch overlap weights with GT mask (vectorized, fast).
    
    Args:
        mask_yx: Original mask (H, W), values > 0 indicate lesion
        input_size: Input image size (after resize+pad)
        patch_size: ViT patch size
        resize_info: (scale, pad_top, pad_left, newH, newW) from resize_keep_ratio_pad
        ring_dilate_pixels: Dilation for ring mask (in original resolution)
        
    Returns:
        lesion_weights: (num_patches,) normalized lesion region weights
        ring_weights: (num_patches,) normalized ring region weights
        
    Note:
        Both weight vectors sum to 1.0 if region is non-empty, else all zeros.
        Output shape is guaranteed: (input_size//patch_size)^2 patches.
    """
    scale, pad_top, pad_left, newH, newW = resize_info
    
    # Resize mask to newH x newW
    mask_tensor = torch.from_numpy(mask_yx.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(newH, newW), mode="nearest").squeeze()
    
    # Pad to input_size
    padH = input_size - newH
    padW = input_size - newW
    pad_bottom = padH - pad_top
    pad_right = padW - pad_left
    
    mask_padded = F.pad(
        mask_resized.unsqueeze(0).unsqueeze(0), 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode="constant", value=0.0
    ).squeeze()
    
    # Compute ring mask (dilate - original) with scaled dilation
    dilate_pixels = min(int(ring_dilate_pixels * scale), 30)  # limit max dilation
    dilated = dilate_mask(mask_padded.numpy(), pixels=dilate_pixels)
    ring_mask = torch.from_numpy((dilated > 0) & (mask_padded.numpy() == 0)).float()
    mask_padded = (mask_padded > 0).float()
    
    # Vectorized patch weight computation via avg_pool2d
    gh = gw = input_size // patch_size
    num_patches = gh * gw
    
    # Lesion weights: fraction of mask > 0 per patch
    lesion_weights = F.avg_pool2d(
        mask_padded.unsqueeze(0).unsqueeze(0), 
        kernel_size=patch_size, 
        stride=patch_size
    ).squeeze().numpy().flatten()
    
    # Ring weights: fraction of ring_mask per patch
    ring_weights = F.avg_pool2d(
        ring_mask.unsqueeze(0).unsqueeze(0), 
        kernel_size=patch_size, 
        stride=patch_size
    ).squeeze().numpy().flatten()
    
    # Normalize to sum=1 (or zeros if empty)
    if lesion_weights.sum() > 0:
        lesion_weights = lesion_weights / lesion_weights.sum()
    if ring_weights.sum() > 0:
        ring_weights = ring_weights / ring_weights.sum()
    
    return lesion_weights.astype(np.float32), ring_weights.astype(np.float32)


@torch.no_grad()
def extract_slice_tokens(
    model,
    slice_yx: np.ndarray,
    input_size: int,
    device: str = "cuda:0"
) -> Tuple[np.ndarray, Tuple[float, int, int, int, int]]:
    """
    Full-slice forward pass, returns patch tokens.
    
    Args:
        model: MedDINOv3 ViT model
        slice_yx: Input image (H, W)
        input_size: Target input size for the model
        device: Torch device
        
    Returns:
        feats_np: Patch token features (N, D) where N = (input_size/patch_size)^2
        resize_info: (scale, pad_top, pad_left, newH, newW) for coordinate mapping
    """
    # Prepare input
    x = torch.from_numpy(slice_yx).float().unsqueeze(0)
    x3 = x.repeat(3, 1, 1)  # grayscale -> 3 channel
    
    # Resize with aspect ratio preservation + padding
    x3, resize_info = _resize_keep_ratio_pad(x3, input_size)
    
    # Forward pass
    out = model(x3.unsqueeze(0).to(device), is_training=True)
    feats = out["x_norm_patchtokens"][0]  # (N, D)
    feats_np = feats.cpu().numpy()
    
    return feats_np, resize_info


def _resize_keep_ratio_pad(x3: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, Tuple]:
    """
    Resize image tensor while preserving aspect ratio, pad to target_size.
    
    Args:
        x3: Input tensor (3, H, W)
        target_size: Target square size
        
    Returns:
        Padded tensor (3, target_size, target_size)
        resize_info: (scale, pad_top, pad_left, newH, newW)
    """
    _, H, W = x3.shape
    
    scale = target_size / max(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    
    x3 = F.interpolate(
        x3.unsqueeze(0),
        size=(newH, newW),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    padH = target_size - newH
    padW = target_size - newW
    
    pad_top = padH // 2
    pad_bottom = padH - pad_top
    pad_left = padW // 2
    pad_right = padW - pad_left
    
    x3 = F.pad(x3, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return x3, (scale, pad_top, pad_left, newH, newW)


def maskpool_embeddings(
    feats_np: np.ndarray,
    lesion_w: np.ndarray,
    ring_w: np.ndarray,
    variant: Literal["lesion", "delta", "lesion+global", "lesion+global+delta"] = "lesion+global+delta",
    include_global: bool = True
) -> np.ndarray:
    """
    Dimension-stable masked pooling with configurable variants.
    
    Args:
        feats_np: Patch token features (N, D)
        lesion_w: Lesion region weights (N,), sum=1 or all zeros
        ring_w: Ring region weights (N,), sum=1 or all zeros
        variant: One of:
            - "lesion": returns D (lesion pooled features)
            - "delta": returns D (lesion - ring, ring missing => zeros)
            - "lesion+global": returns 2D (concat lesion, global)
            - "lesion+global+delta": returns 3D (concat lesion, global, delta)
        include_global: If True, include global in global-variants (for API compat)
        
    Returns:
        emb_np: Embedding of shape (D,), (2D,), or (3D,) depending on variant.
                DIMENSION IS STABLE: empty lesion/ring filled with zeros.
                
    Note:
        - If lesion_w.sum() == 0, e_lesion falls back to mean(feats, axis=0)
        - If ring_w.sum() == 0, e_ring falls back to mean(feats, axis=0)
        - Global is always mean(feats, axis=0)
    """
    N, D = feats_np.shape
    
    # Compute component embeddings
    e_global = feats_np.mean(axis=0)  # (D,) - always valid
    
    # Lesion embedding (fallback to global pooling if mask is empty to avoid zero vector)
    if lesion_w.sum() > 0:
        e_lesion = (feats_np * lesion_w[:, np.newaxis]).sum(axis=0)
    else:
        e_lesion = feats_np.mean(axis=0).astype(np.float32)
    
    # Ring embedding (fallback to global pooling if ring is empty)
    if ring_w.sum() > 0:
        e_ring = (feats_np * ring_w[:, np.newaxis]).sum(axis=0)
    else:
        e_ring = feats_np.mean(axis=0).astype(np.float32)
    
    # Delta embedding (lesion - ring)
    e_delta = e_lesion - e_ring
    
    # Delta_global embedding (lesion - global) - for ablation
    e_delta_global = e_lesion - e_global
    
    # Assemble output based on variant
    if variant == "lesion":
        return e_lesion.astype(np.float32)
    
    elif variant == "delta":
        return e_delta.astype(np.float32)
    
    elif variant == "delta_global":
        # For ablation: compare ring context vs global context
        return e_delta_global.astype(np.float32)
    
    elif variant == "lesion+global":
        return np.concatenate([e_lesion, e_global]).astype(np.float32)
    
    elif variant == "lesion+ring":
        # For learnable delta: concat(lesion, ring) so classifier can compute lesion - α*ring
        return np.concatenate([e_lesion, e_ring]).astype(np.float32)
    
    elif variant == "lesion+global+delta":
        return np.concatenate([e_lesion, e_global, e_delta]).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Supported: {list(MASKPOOL_VARIANTS.keys())}")


def get_variant_dim_multiplier(variant: str) -> int:
    """
    Get output dimension multiplier for a variant.
    
    Args:
        variant: One of MASKPOOL_VARIANTS keys
        
    Returns:
        Multiplier (1, 2, or 3) to multiply with base feature dim D
    """
    if variant not in MASKPOOL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Supported: {list(MASKPOOL_VARIANTS.keys())}")
    return MASKPOOL_VARIANTS[variant]


def match_dim_embedding(
    embedding: np.ndarray,
    target_variant: str,
    base_dim: int
) -> np.ndarray:
    """
    Tile/replicate a 1D embedding to match target variant dimension.
    
    Useful for baselines (full_global, crop_roi) to output matched dimensions.
    
    Args:
        embedding: Input embedding (D,)
        target_variant: Target variant name (e.g., "lesion+global+delta")
        base_dim: Base feature dimension D
        
    Returns:
        Tiled embedding matching target variant dimension
        
    Example:
        If target_variant="lesion+global+delta" (3D) and input is (D,),
        returns concat([embedding, embedding, embedding]) = (3D,)
    """
    multiplier = get_variant_dim_multiplier(target_variant)
    
    if len(embedding) == base_dim:
        # Need to tile
        return np.tile(embedding, multiplier).astype(np.float32)
    elif len(embedding) == base_dim * multiplier:
        # Already correct dimension
        return embedding.astype(np.float32)
    else:
        raise ValueError(f"Embedding dim {len(embedding)} doesn't match "
                        f"base_dim={base_dim} or target {base_dim * multiplier}")


