#!/usr/bin/env python3
"""
Example script demonstrating the Oracle Upper Bound method usage.

This example shows how to:
1. Configure the method
2. Run a single experiment
3. Access and interpret results
"""

import os
from oracle_upper_bound_v2 import run_experiment, DEFAULT_CONFIG, print_summary

def main():
    print("=" * 70)
    print("MedDINOv3 Inference - Oracle Upper Bound Example")
    print("=" * 70)
    
    # Configure data path
    # Update this to point to your preprocessed data directory
    PREPROCESSED_ROOT = "/path/to/your/preprocessed/data"
    
    # Example 1: Run with default configuration
    print("\n[Example 1] Running with default configuration...")
    print("-" * 70)
    
    results = run_experiment(
        # Data path (you may need to modify oracle_upper_bound.py if using different path)
        # This is typically set as a global variable in the module
        
        # Backbone selection
        backbone="meddino",  # or "dinov3_hf"
        
        # Feature extraction mode
        mode="full_maskpool",  # Main method
        maskpool_variant="delta",  # Context-aware delta readout
        rim_width=20,  # Ring dilation width
        
        # Slice selection
        selection_strategy="topk",
        K=3,  # Select top 3 slices
        aggregation="mean",  # Mean aggregation
        
        # Classifier
        classifier_type="mlp",
        hidden_dims=[128, 64],
        dropout=0.75,
        
        # Training
        epochs=200,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-2,
        patience=30,
        n_folds=5,  # 5-fold cross-validation
        
        # I/O
        use_cache=True,  # Enable feature caching
        cache_dir="./feature_cache",
        out_root="./example_outputs",
        extra_tag="example_run",
    )
    
    # Print results summary
    print_summary(results)
    
    # Access detailed results
    cv_results = results["cv_results"]
    print(f"\nDetailed Results:")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"  Overall Accuracy: {cv_results['overall_accuracy']:.4f}")
    print(f"  Overall AUC: {cv_results['overall_auc']:.4f}")
    print(f"  Overfit Gap: {cv_results['overfit_gap']:.4f}")
    
    # Example 2: Compare different variants
    print("\n" + "=" * 70)
    print("[Example 2] Comparing different maskpool variants...")
    print("-" * 70)
    
    variants = ["lesion", "delta", "lesion+global", "lesion+global+delta"]
    
    for variant in variants:
        print(f"\nTesting variant: {variant}")
        results = run_experiment(
            mode="full_maskpool",
            maskpool_variant=variant,
            K=3,
            out_root="./example_outputs",
            extra_tag=f"variant_{variant}",
            use_cache=True,  # Reuse cached features
        )
        cv = results["cv_results"]
        print(f"  AUC: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
    
    # Example 3: Different slice selection strategies
    print("\n" + "=" * 70)
    print("[Example 3] Comparing slice selection strategies...")
    print("-" * 70)
    
    strategies = ["topk", "topk_unique", "window"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        results = run_experiment(
            selection_strategy=strategy,
            K=5,
            out_root="./example_outputs",
            extra_tag=f"strategy_{strategy}",
            use_cache=True,
        )
        cv = results["cv_results"]
        print(f"  AUC: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nOutput files are saved in ./example_outputs/")
    print("Feature cache is saved in ./feature_cache/")


if __name__ == "__main__":
    main()

