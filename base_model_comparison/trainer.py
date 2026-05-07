"""
Training and Evaluation Pipeline for BCI Transfer Learning.

Includes:
- Pretraining on source subjects
- Fine-tuning / domain adaptation on target subject
- Evaluation with accuracy, kappa, confusion matrix
- Early stopping and learning rate scheduling
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, classification_report,
)
from typing import Dict, Optional, Tuple, List

from base_model_comparison.transfer_learning import (
    DomainAdaptationModel, DANN, ProgressiveNet, MultiSourceTransfer,
)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ──────────────────────────────────────────────────────────────────────
# Standard Training and Evaluation
# ──────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type=device.split(":")[0]):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    total = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    return total_loss / total, acc, all_preds, all_labels


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute classification metrics."""
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ──────────────────────────────────────────────────────────────────────
# Source Pretraining
# ──────────────────────────────────────────────────────────────────────

def pretrain_on_source(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    val_loader: Optional[DataLoader] = None,
    save_path: Optional[str] = None,
) -> nn.Module:
    """
    Pretrain model on source subject(s) data.

    Args:
        model: Model to train
        train_loader: Source training data
        config: TrainConfig
        val_loader: Optional validation loader
        save_path: Path to save best model checkpoint
    """
    device = config.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_pretrain, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs_pretrain, eta_min=1e-6
    )
    scaler = GradScaler(device) if config.use_amp and "cuda" in device else None
    early_stop = EarlyStopping(patience=config.patience)

    print(f"\n{'='*60}")
    print(f"Source Pretraining ({config.n_epochs_pretrain} epochs)")
    print(f"{'='*60}")

    best_val_acc = 0.0
    for epoch in range(config.n_epochs_pretrain):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, config.max_grad_norm
        )
        scheduler.step()

        log_msg = f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"

        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            early_stop(val_loss, model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            early_stop(train_loss, model)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(log_msg)

        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model


# ──────────────────────────────────────────────────────────────────────
# Fine-Tuning on Target
# ──────────────────────────────────────────────────────────────────────

def finetune_on_target(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config,
    freeze_features: bool = True,
    n_classes: Optional[int] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tune pretrained model on target subject calibration data.

    Args:
        model: Pretrained model
        train_loader: Target calibration data
        test_loader: Target test data
        config: TrainConfig
        freeze_features: If True, only train classifier head
        n_classes: Target number of classes
    """
    device = config.device
    model = model.to(device)

    # Replace classifier if needed
    if n_classes is not None:
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                if module.out_features != n_classes:
                    in_features = module.in_features
                    model.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in_features, n_classes),
                    ).to(device)
                break

    # Freeze features if specified
    if freeze_features and hasattr(model, "feature_extractor"):
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        param_groups = [{"params": model.classifier.parameters(), "lr": config.lr_finetune}]
    else:
        param_groups = [
            {"params": model.feature_extractor.parameters(), "lr": config.lr_finetune * 0.1},
            {"params": model.classifier.parameters(), "lr": config.lr_finetune},
        ]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs_finetune, eta_min=1e-7
    )
    early_stop = EarlyStopping(patience=config.patience)

    print(f"\n{'='*60}")
    print(f"Target Fine-tuning ({config.n_epochs_finetune} epochs, freeze={freeze_features})")
    print(f"{'='*60}")

    for epoch in range(config.n_epochs_finetune):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm=config.max_grad_norm
        )
        scheduler.step()

        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Test: {val_loss:.4f}/{val_acc:.4f}")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)

    # Final evaluation
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(preds, labels)

    return model, metrics


# ──────────────────────────────────────────────────────────────────────
# Domain Adaptation Training
# ──────────────────────────────────────────────────────────────────────

def train_domain_adaptation(
    model: DomainAdaptationModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    test_loader: DataLoader,
    config,
    domain_loss_weight: float = 0.5,
) -> Tuple[nn.Module, Dict]:
    """
    Train with domain adaptation (MMD/CORAL).

    The model minimizes: L = L_cls(source) + λ * L_domain(source, target)
    """
    device = config.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_finetune, weight_decay=config.weight_decay)
    early_stop = EarlyStopping(patience=config.patience)

    print(f"\n{'='*60}")
    print(f"Domain Adaptation Training ({model.adaptation_type.upper()}, λ={domain_loss_weight})")
    print(f"{'='*60}")

    target_iter = iter(target_loader)

    for epoch in range(config.n_epochs_finetune):
        model.train()
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        n_batches = 0

        for source_batch in source_loader:
            # Get target batch (cycle if exhausted)
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            source_x = source_batch[0].to(device)
            source_y = source_batch[1].to(device)
            target_x = target_batch[0].to(device)

            optimizer.zero_grad()
            source_logits, domain_loss = model(source_x, target_x)
            cls_loss = criterion(source_logits, source_y)
            total_loss = cls_loss + domain_loss_weight * domain_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_dom_loss += domain_loss.item()
            n_batches += 1

        avg_cls = total_cls_loss / n_batches
        avg_dom = total_dom_loss / n_batches

        # Evaluate on target test
        val_loss, val_acc, _, _ = evaluate_domain_model(model, test_loader, criterion, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Cls: {avg_cls:.4f} | Dom: {avg_dom:.4f} | "
                  f"Target Acc: {val_acc:.4f}")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)

    _, _, preds, labels = evaluate_domain_model(model, test_loader, criterion, device)
    metrics = compute_metrics(preds, labels)
    return model, metrics


@torch.no_grad()
def evaluate_domain_model(model, loader, criterion, device):
    """Evaluate a DomainAdaptationModel or DANN using predict()."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model.predict(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(all_labels), acc, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────
# DANN Training
# ──────────────────────────────────────────────────────────────────────

def train_dann(
    model: DANN,
    source_loader: DataLoader,
    target_loader: DataLoader,
    test_loader: DataLoader,
    config,
) -> Tuple[nn.Module, Dict]:
    """
    Train DANN with gradient reversal for domain-adversarial adaptation.
    Alpha (reversal strength) increases from 0 to 1 during training.
    """
    device = config.device
    model = model.to(device)
    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_finetune, weight_decay=config.weight_decay)
    early_stop = EarlyStopping(patience=config.patience)

    print(f"\n{'='*60}")
    print(f"DANN Training")
    print(f"{'='*60}")

    target_iter = iter(target_loader)
    total_steps = config.n_epochs_finetune * len(source_loader)

    for epoch in range(config.n_epochs_finetune):
        model.train()
        for step, source_batch in enumerate(source_loader):
            # Progressive alpha schedule
            p = (epoch * len(source_loader) + step) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            source_x = source_batch[0].to(device)
            source_y = source_batch[1].to(device)
            target_x = target_batch[0].to(device)

            optimizer.zero_grad()

            # Source: classification + domain
            src_cls_logits, src_dom_logits = model(source_x, alpha)
            src_cls_loss = cls_criterion(src_cls_logits, source_y)
            src_dom_labels = torch.zeros(source_x.size(0), dtype=torch.long, device=device)
            src_dom_loss = dom_criterion(src_dom_logits, src_dom_labels)

            # Target: domain only
            tgt_cls_logits, tgt_dom_logits = model(target_x, alpha)
            tgt_dom_labels = torch.ones(target_x.size(0), dtype=torch.long, device=device)
            tgt_dom_loss = dom_criterion(tgt_dom_logits, tgt_dom_labels)

            loss = src_cls_loss + src_dom_loss + tgt_dom_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        val_loss, val_acc, _, _ = evaluate_domain_model(model, test_loader, cls_criterion, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | alpha={alpha:.3f} | Target Acc: {val_acc:.4f}")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)
    _, _, preds, labels = evaluate_domain_model(model, test_loader, cls_criterion, device)
    metrics = compute_metrics(preds, labels)
    return model, metrics


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline Runner
# ──────────────────────────────────────────────────────────────────────

def run_transfer_pipeline(
    model: nn.Module,
    source_loader: DataLoader,
    target_train_loader: DataLoader,
    target_test_loader: DataLoader,
    train_config,
    transfer_config,
    n_classes: int,
    source_models: Optional[List[nn.Module]] = None,
) -> Dict:
    """
    Run complete TL pipeline: pretrain → transfer → evaluate.

    Args:
        model: Base model architecture
        source_loader: Source subjects data
        target_train_loader: Target calibration data
        target_test_loader: Target test data
        train_config: TrainConfig
        transfer_config: TransferConfig
        n_classes: Number of classes
        source_models: List of pre-trained per-source models (for multi_source)
    Returns:
        results: Dictionary with metrics and model
    """
    from base_model_comparison.transfer_learning import TransferLearningManager

    device = train_config.device
    strategy = transfer_config.strategy

    results = {"strategy": strategy}

    # Step 1: Pretrain on source data (for all strategies except multi_source)
    if strategy != "multi_source":
        print("\n[Step 1] Pretraining on source subjects...")
        pretrained_model = pretrain_on_source(
            model, source_loader, train_config,
            save_path=f"checkpoints/pretrained_{strategy}.pt",
        )
    else:
        pretrained_model = model

    # Step 2: Apply transfer learning strategy
    print(f"\n[Step 2] Applying TL strategy: {strategy}")

    if strategy in ("fine_tune", "feature_extract"):
        freeze = strategy == "feature_extract" or transfer_config.freeze_feature_extractor
        final_model, metrics = finetune_on_target(
            pretrained_model, target_train_loader, target_test_loader,
            train_config, freeze_features=freeze, n_classes=n_classes,
        )

    elif strategy == "domain_adapt":
        from base_model_comparison.transfer_learning import DomainAdaptationModel
        da_model = DomainAdaptationModel(pretrained_model, adaptation_type="mmd")
        final_model, metrics = train_domain_adaptation(
            da_model, source_loader, target_train_loader, target_test_loader,
            train_config, domain_loss_weight=transfer_config.domain_loss_weight,
        )

    elif strategy == "dann":
        from base_model_comparison.transfer_learning import DANN
        dann_model = DANN(pretrained_model, n_classes)
        final_model, metrics = train_dann(
            dann_model, source_loader, target_train_loader, target_test_loader, train_config,
        )

    elif strategy == "progressive":
        from base_model_comparison.transfer_learning import ProgressiveNet
        prog_model = ProgressiveNet(pretrained_model, n_classes)
        final_model, metrics = finetune_on_target(
            prog_model, target_train_loader, target_test_loader,
            train_config, freeze_features=False, n_classes=n_classes,
        )

    elif strategy == "multi_source":
        if source_models is None:
            raise ValueError("multi_source strategy requires source_models list")
        from base_model_comparison.transfer_learning import MultiSourceTransfer
        ms_model = MultiSourceTransfer(source_models, n_classes)
        final_model, metrics = finetune_on_target(
            ms_model, target_train_loader, target_test_loader,
            train_config, freeze_features=False, n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results["metrics"] = metrics
    results["model"] = final_model

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {strategy}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Kappa:    {metrics['kappa']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    return results
