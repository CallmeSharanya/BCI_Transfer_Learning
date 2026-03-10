"""
Main entry point for BCI Transfer Learning experiments.

Usage:
    python main.py --dataset bci_iv_2a --model eegnet --strategy fine_tune --target 9
    python main.py --dataset physionet --model deepconvnet --strategy dann --target 5
    python main.py --compare-all --dataset bci_iv_2a --target 9
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List

from config import (
    PreprocessingConfig, DataConfig, ModelConfig, TransferConfig, TrainConfig,
)
from models import build_model
from datasets import create_source_target_loaders, load_and_preprocess_subject
from trainer import run_transfer_pipeline, pretrain_on_source, finetune_on_target


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataset_params(dataset_name: str, data_config: DataConfig) -> Dict:
    """Get dataset-specific parameters."""
    params = {
        "bci_iv_2a": {
            "data_dir": data_config.bci_iv_2a_path,
            "n_channels": 22,
            "n_classes": 4,
            "all_subjects": list(range(1, 10)),
            "srate": 250.0,
            "class_names": ["Left Hand", "Right Hand", "Both Feet", "Tongue"],
        },
        "bci_iv_2b": {
            "data_dir": data_config.bci_iv_2b_path,
            "n_channels": 3,
            "n_classes": 2,
            "all_subjects": list(range(1, 10)),
            "srate": 250.0,
            "class_names": ["Left Hand", "Right Hand"],
        },
        "physionet": {
            "data_dir": data_config.physionet_path,
            "n_channels": 64,
            "n_classes": 4,
            "all_subjects": list(range(1, 110)),
            "srate": 160.0,
            "class_names": ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        },
    }
    if dataset_name not in params:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(params.keys())}")
    return params[dataset_name]


def run_single_experiment(
    dataset_name: str,
    model_type: str,
    strategy: str,
    target_subject: int,
    source_subjects: List[int],
    preproc_config: PreprocessingConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    transfer_config: TransferConfig,
    train_config: TrainConfig,
) -> Dict:
    """Run a single transfer learning experiment."""
    set_seed(train_config.seed)
    device = train_config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        train_config.device = "cpu"
        train_config.use_amp = False

    ds_params = get_dataset_params(dataset_name, data_config)
    n_channels = ds_params["n_channels"]
    n_classes = ds_params["n_classes"]
    n_samples = int((preproc_config.tmax - preproc_config.tmin) * preproc_config.target_srate)

    print(f"\n{'#'*70}")
    print(f"# Experiment: {dataset_name} | {model_type} | {strategy}")
    print(f"# Target: Subject {target_subject} | Source: {source_subjects}")
    print(f"# Channels: {n_channels} | Samples: {n_samples} | Classes: {n_classes}")
    print(f"# Device: {device}")
    print(f"{'#'*70}")

    # Create data loaders
    source_loader, target_train_loader, target_test_loader = create_source_target_loaders(
        dataset_name=dataset_name,
        data_dir=ds_params["data_dir"],
        source_subjects=source_subjects,
        target_subject=target_subject,
        preproc_config=preproc_config,
        train_config=train_config,
        n_target_trials_per_class=transfer_config.n_target_trials_per_class,
        apply_ea=transfer_config.use_euclidean_alignment,
    )

    # Build model
    model = build_model(model_type, n_channels, n_samples, n_classes, model_config)
    print(f"\nModel: {model_type} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Run transfer pipeline
    transfer_config.strategy = strategy
    results = run_transfer_pipeline(
        model=model,
        source_loader=source_loader,
        target_train_loader=target_train_loader,
        target_test_loader=target_test_loader,
        train_config=train_config,
        transfer_config=transfer_config,
        n_classes=n_classes,
    )

    return results


def compare_all_strategies(
    dataset_name: str,
    model_type: str,
    target_subject: int,
    source_subjects: List[int],
    preproc_config: PreprocessingConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    transfer_config: TransferConfig,
    train_config: TrainConfig,
) -> Dict:
    """Compare all TL strategies on the same dataset/subject."""
    strategies = ["fine_tune", "feature_extract", "domain_adapt", "dann", "progressive"]
    all_results = {}

    for strategy in strategies:
        print(f"\n\n{'*'*70}")
        print(f"* Strategy: {strategy}")
        print(f"{'*'*70}")
        try:
            results = run_single_experiment(
                dataset_name, model_type, strategy, target_subject,
                source_subjects, preproc_config, data_config, model_config,
                transfer_config, train_config,
            )
            all_results[strategy] = {
                "accuracy": results["metrics"]["accuracy"],
                "kappa": results["metrics"]["kappa"],
            }
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            all_results[strategy] = {"accuracy": 0.0, "kappa": 0.0, "error": str(e)}

    # Summary
    print(f"\n\n{'='*70}")
    print(f"COMPARISON SUMMARY: {dataset_name} | {model_type} | Target Subject {target_subject}")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Kappa':>10}")
    print(f"{'-'*40}")
    for strategy, res in sorted(all_results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        print(f"{strategy:<20} {res.get('accuracy', 0):.4f}     {res.get('kappa', 0):.4f}")

    return all_results


def cross_subject_evaluation(
    dataset_name: str,
    model_type: str,
    strategy: str,
    preproc_config: PreprocessingConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    transfer_config: TransferConfig,
    train_config: TrainConfig,
) -> Dict:
    """Leave-one-subject-out cross-validation across all subjects."""
    ds_params = get_dataset_params(dataset_name, data_config)
    all_subjects = ds_params["all_subjects"]
    subject_results = {}

    for target_subject in all_subjects:
        source_subjects = [s for s in all_subjects if s != target_subject]
        try:
            results = run_single_experiment(
                dataset_name, model_type, strategy, target_subject,
                source_subjects, preproc_config, data_config, model_config,
                transfer_config, train_config,
            )
            subject_results[target_subject] = {
                "accuracy": results["metrics"]["accuracy"],
                "kappa": results["metrics"]["kappa"],
            }
        except Exception as e:
            print(f"Subject {target_subject} failed: {e}")
            subject_results[target_subject] = {"accuracy": 0.0, "kappa": 0.0}

    # Summary
    accs = [r["accuracy"] for r in subject_results.values()]
    kappas = [r["kappa"] for r in subject_results.values()]
    print(f"\n{'='*70}")
    print(f"CROSS-SUBJECT RESULTS: {dataset_name} | {model_type} | {strategy}")
    print(f"{'='*70}")
    for sid, res in subject_results.items():
        print(f"  Subject {sid:2d}: Acc={res['accuracy']:.4f}, Kappa={res['kappa']:.4f}")
    print(f"\n  Mean Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

    return subject_results


def main():
    parser = argparse.ArgumentParser(description="BCI Transfer Learning for Motor Imagery")
    parser.add_argument("--dataset", type=str, default="bci_iv_2a",
                        choices=["bci_iv_2a", "bci_iv_2b", "physionet"],
                        help="Dataset to use")
    parser.add_argument("--model", type=str, default="eegnet",
                        choices=["eegnet", "shallowconvnet", "deepconvnet"],
                        help="Model architecture")
    parser.add_argument("--strategy", type=str, default="fine_tune",
                        choices=["fine_tune", "feature_extract", "domain_adapt",
                                 "dann", "progressive", "multi_source"],
                        help="Transfer learning strategy")
    parser.add_argument("--target", type=int, default=9,
                        help="Target subject ID")
    parser.add_argument("--source", type=int, nargs="*", default=None,
                        help="Source subject IDs (default: all except target)")
    parser.add_argument("--n-cal", type=int, default=10,
                        help="Number of calibration trials per class for target")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all TL strategies")
    parser.add_argument("--cross-subject", action="store_true",
                        help="Run leave-one-subject-out evaluation")
    parser.add_argument("--epochs-pretrain", type=int, default=200)
    parser.add_argument("--epochs-finetune", type=int, default=100)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ea", action="store_true",
                        help="Disable Euclidean Alignment")
    parser.add_argument("--no-ica", action="store_true",
                        help="Disable ICA artifact removal")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root data directory")

    args = parser.parse_args()

    # Build configs
    preproc_config = PreprocessingConfig(
        use_ica=not args.no_ica,
    )
    data_config = DataConfig(
        bci_iv_2a_path=os.path.join(args.data_dir, "bci_iv_2a"),
        bci_iv_2b_path=os.path.join(args.data_dir, "bci_iv_2b"),
        physionet_path=os.path.join(args.data_dir, "physionet_mi"),
    )
    model_config = ModelConfig(model_type=args.model)
    transfer_config = TransferConfig(
        strategy=args.strategy,
        target_subject=args.target,
        n_target_trials_per_class=args.n_cal,
        use_euclidean_alignment=not args.no_ea,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        n_epochs_pretrain=args.epochs_pretrain,
        n_epochs_finetune=args.epochs_finetune,
        lr_pretrain=args.lr_pretrain,
        lr_finetune=args.lr_finetune,
        seed=args.seed,
        device=args.device,
    )

    # Determine source subjects
    ds_params = get_dataset_params(args.dataset, data_config)
    if args.source is not None:
        source_subjects = args.source
    else:
        source_subjects = [s for s in ds_params["all_subjects"] if s != args.target]

    transfer_config.source_subjects = source_subjects

    # Run experiment
    if args.compare_all:
        compare_all_strategies(
            args.dataset, args.model, args.target, source_subjects,
            preproc_config, data_config, model_config, transfer_config, train_config,
        )
    elif args.cross_subject:
        cross_subject_evaluation(
            args.dataset, args.model, args.strategy,
            preproc_config, data_config, model_config, transfer_config, train_config,
        )
    else:
        run_single_experiment(
            args.dataset, args.model, args.strategy, args.target, source_subjects,
            preproc_config, data_config, model_config, transfer_config, train_config,
        )


if __name__ == "__main__":
    main()
