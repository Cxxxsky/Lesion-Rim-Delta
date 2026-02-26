#!/usr/bin/env python3
"""
================================================================================
Oracle Upper Bound v2 — Config-based Clean API
================================================================================

Thin wrapper around oracle_upper_bound.py that provides:
1. Unified Config dict instead of scattered global variables
2. Single run_experiment(cfg) call instead of 20+ parameter passing
3. Single backbone switch via cfg["backbone"] instead of multiple places

All core computation logic (model building, feature extraction, training, evaluation)
is reused from oracle_upper_bound.py. This file only handles parameter management.

Usage:
    # Run single experiment directly
    python oracle_upper_bound_v2.py

    # Use as library
    from oracle_upper_bound_v2 import run_experiment, DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, "backbone": "dinov3_hf", "K": 5}
    results = run_experiment(cfg)

Author: YYX Research
================================================================================
"""

import os
import copy
from typing import Dict, Any, Optional

# Reuse all core functions from v1 (don't reinvent the wheel)
import oracle_upper_bound as _oub
from oracle_upper_bound import (
    # Data
    discover_all_cases,
    # Model
    build_model, build_meddino_vitb, build_dinov3_hf,
    get_input_size,
    # Feature & Training
    train_eval_oracle_cv,
    # Utilities
    get_experiment_dir,
    PREPROCESSED_ROOT,
)


# =============================================================================
# Default Configuration — All parameters centralized here
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # ----- Backbone -----
    # "meddino"  : MedDINOv3 ViT-B-16 (768-dim, CT pretrained, input=2048)
    # "dinov3_hf": DINOv3 ViT-H+-16  (1280-dim, natural image pretrained, input=518)
    "backbone": "meddino",

    # ----- Feature Extraction -----
    "mode": "full_maskpool",          # "full_global" / "full_maskpool" / "crop_roi"
    "maskpool_variant": "delta",      # "lesion" / "delta" / "lesion+global" / "lesion+global+delta" / "lesion+ring"
    "match_dim_to_variant": False,    # True → tile global/crop output to match variant dim

    # ----- Mask Pooling -----
    "rim_width": 20,                  # Ring region width (pixels), default matches RING_DILATE_PIXELS

    # ----- Slice Selection -----
    "selection_strategy": "topk",
    "K": 3,
    "aggregation": "mean",            # "mean" / "weighted" / "max" (ignored for MIL)

    # ----- Classifier -----
    # Non-MIL: "mlp", "linear", "learnable_delta"
    # MIL:     "attention_mil", "gated_attention_mil", "transmil"
    "classifier_type": "mlp",
    "hidden_dims": [128, 64],
    "dropout": 0.75,
    "alpha_init": 1.0,               # Only for learnable_delta
    "alpha_mode": "scalar",           # "scalar" / "vector", only for learnable_delta

    # ----- Training -----
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "patience": 30,
    "n_folds": 5,
    "seed": 42,

    # ----- I/O -----
    "use_cache": True,
    "cache_dir": "./feature_cache",
    "out_root": "./oracle_outputs_v2",  # Root directory for all experiments
    "extra_tag": "",                    # Optional suffix to distinguish experiments with same config

    # ----- Mask Source -----
    # None = Oracle (GT mask from PREPROCESSED_ROOT)
    # Directory path = Predicted mask (should contain {case_id}_seg.b2nd files)
    "seg_dir": None,

    # ----- Bbox Mask -----
    # True = Degrade precise mask to its bounding-box rectangle (for ablation)
    "use_bbox_mask": False,
}


# =============================================================================
# Core Entry Point
# =============================================================================

def make_config(**overrides) -> Dict[str, Any]:
    """
    Override default config with overrides, return new dict.

    Example:
        cfg = make_config(backbone="dinov3_hf", K=5, extra_tag="test")
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"Unknown config key: '{k}'. Valid keys: {sorted(cfg.keys())}")
        cfg[k] = v
    return cfg


def _resolve_out_dir(cfg: Dict[str, Any]) -> str:
    """Generate clean output directory path from config (no nesting)."""
    # get_experiment_dir returns OUT_DIR_BASE/oracle_xxx, here we override OUT_DIR_BASE
    _oub.OUT_DIR_BASE = cfg["out_root"]
    return get_experiment_dir(
        mode=cfg["mode"],
        classifier_type=cfg["classifier_type"],
        K=cfg["K"],
        dropout=cfg["dropout"],
        aggregation=cfg["aggregation"],
        extra_tag=cfg["extra_tag"],
        backbone=cfg["backbone"],
    )


def _sync_globals(cfg: Dict[str, Any]):
    """Sync config to oracle_upper_bound module-level global variables."""
    _oub.BACKBONE = cfg["backbone"]
    _oub.INPUT_SIZE = get_input_size(cfg["backbone"])
    _oub.CACHE_DIR = cfg["cache_dir"]
    _oub.OUT_DIR_BASE = cfg["out_root"]
    _oub.SEG_DIR = cfg.get("seg_dir", None)
    _oub.USE_BBOX_MASK = cfg.get("use_bbox_mask", False)


def run_experiment(
    cfg: Optional[Dict[str, Any]] = None,
    model=None,
    model_info: Optional[Dict] = None,
    case_list=None,
    **overrides,
) -> Dict:
    """
    Run a complete Oracle Upper Bound experiment.

    Args:
        cfg:        Full config dict (None → use DEFAULT_CONFIG)
        model:      Pre-built model (None → auto-build)
        model_info: Info dict returned by build_model() (paired with model)
        case_list:  Case list (None → auto-discover)
        **overrides: Directly override fields in cfg

    Returns:
        results dict (consistent with train_eval_oracle_cv)

    Example:
        # Simplest usage — default MedDINOv3
        results = run_experiment()

        # Switch backbone
        results = run_experiment(backbone="dinov3_hf")

        # Pass existing model (avoid reloading)
        model, info = build_model("dinov3_hf")
        r1 = run_experiment(model=model, model_info=info, backbone="dinov3_hf", K=3)
        r2 = run_experiment(model=model, model_info=info, backbone="dinov3_hf", K=5)
    """
    # 1. Merge config
    if cfg is None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    else:
        cfg = copy.deepcopy(cfg)
    cfg.update(overrides)

    # 2. Sync global variables
    _sync_globals(cfg)

    # 3. Auto-discover cases
    if case_list is None:
        case_list = discover_all_cases(PREPROCESSED_ROOT)

    n_ame = sum(1 for _, l in case_list if l == 1)
    n_koc = sum(1 for _, l in case_list if l == 0)
    print(f"[Dataset] {len(case_list)} cases (AME={n_ame}, KOC={n_koc})")

    # 4. Build model (if not provided)
    if model is None or model_info is None:
        print(f"[Model] Building {cfg['backbone']}...")
        model, model_info = build_model(cfg["backbone"])
    print(f"[Model] backbone={model_info['backbone']}, "
          f"hidden_size={model_info['hidden_size']}, "
          f"patch_size={model_info['patch_size']}")

    # 5. Generate output directory
    out_dir = _resolve_out_dir(cfg)
    print(f"[Output] {out_dir}")

    # 6. Run experiment
    results = train_eval_oracle_cv(
        case_list=case_list,
        model=model,
        model_info=model_info,
        selection_strategy=cfg["selection_strategy"],
        K=cfg["K"],
        mode=cfg["mode"],
        aggregation=cfg["aggregation"],
        maskpool_variant=cfg["maskpool_variant"],
        match_dim_to_variant=cfg["match_dim_to_variant"],
        rim_width=cfg.get("rim_width", 20),
        n_folds=cfg["n_folds"],
        classifier_type=cfg["classifier_type"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        alpha_init=cfg["alpha_init"],
        alpha_mode=cfg["alpha_mode"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        seed=cfg["seed"],
        use_cache=cfg["use_cache"],
        out_dir=out_dir,
        verbose=True,
    )

    return results


def print_summary(results: Dict):
    """Print experiment results summary."""
    cv = results.get("cv_results", {})
    cfg = results.get("config", {})

    print(f"\n{'='*60}")
    print(f"  Backbone : {cfg.get('backbone', '?')}")
    print(f"  Mode     : {cfg.get('mode', '?')}")
    print(f"  Variant  : {cfg.get('maskpool_variant', '?')}")
    print(f"  K        : {cfg.get('K', '?')}")
    print(f"  Classifier: {cfg.get('classifier_type', '?')}")
    print(f"  ---")
    print(f"  Accuracy : {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}")
    print(f"  AUC      : {cv.get('mean_auc', 0):.4f} ± {cv.get('std_auc', 0):.4f}")
    print(f"  Overall  : Acc={cv.get('overall_accuracy', 0):.4f}, AUC={cv.get('overall_auc', 0):.4f}")
    print(f"  Overfit  : {cv.get('overfit_gap', 0):.4f}")
    if cv.get("mean_alpha") is not None:
        print(f"  Alpha    : {cv['mean_alpha']:.4f}")
    print(f"{'='*60}")


# =============================================================================
# Main Function — Run as standalone script
# =============================================================================

def main():
    """
    Usage: python oracle_upper_bound_v2.py

    To modify config? Just edit the overrides dict below, clear and simple.
    """
    print("\n" + "=" * 70)
    print("Oracle Upper Bound v2 (Config-based)")
    print("=" * 70 + "\n")

    # =========================================================================
    # Modify experiment config here — only write fields that override defaults
    # =========================================================================
    results = run_experiment(
        # ---- Backbone ----
        backbone="meddino",        # "meddino" / "dinov3_hf"

        # ---- Feature ----
        mode="full_maskpool",
        maskpool_variant="lesion+global+delta",
        match_dim_to_variant=False,

        # ---- Selection ----
        K=3,
        aggregation="weighted",

        # ---- Classifier ----
        classifier_type="mlp",
        dropout=0.75,

        # ---- I/O ----
        out_root="./oracle_outputs_v2",
        extra_tag="T1_ours_varL+G+D",
        use_cache=True,
    )

    print_summary(results)
    print("[DONE]")


if __name__ == "__main__":
    main()

