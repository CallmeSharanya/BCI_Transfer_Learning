"""
Transfer Learning Strategies for BCI Motor Imagery.

Implements five TL approaches:
1. Fine-Tuning: Pretrain on source, fine-tune classifier (or all layers) on target
2. Feature Extraction: Freeze pretrained feature extractor, train new classifier
3. Domain Adaptation: MMD / CORAL loss to align source and target distributions
4. Progressive Neural Networks: Lateral connections from frozen source model
5. Multi-Source Transfer: Weighted combination of multiple source models
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Tuple


# ──────────────────────────────────────────────────────────────────────
# Domain Adaptation Losses
# ──────────────────────────────────────────────────────────────────────

def compute_mmd(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) with Gaussian kernel.
    Measures distance between source and target feature distributions.

    MMD^2 = E[k(xs,xs')] + E[k(xt,xt')] - 2*E[k(xs,xt)]
    """
    def gaussian_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    n_s = source_features.size(0)
    n_t = target_features.size(0)

    # Multi-kernel MMD with multiple bandwidths for robustness
    mmd = torch.tensor(0.0, device=source_features.device)
    for sigma in [0.1, 0.5, 1.0, 5.0, 10.0]:
        k_ss = gaussian_kernel(source_features, source_features, sigma)
        k_tt = gaussian_kernel(target_features, target_features, sigma)
        k_st = gaussian_kernel(source_features, target_features, sigma)

        mmd += (k_ss.sum() / (n_s * n_s) + k_tt.sum() / (n_t * n_t)
                - 2 * k_st.sum() / (n_s * n_t))

    return mmd / 5.0


def compute_coral(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    CORAL (CORrelation ALignment) loss.
    Aligns second-order statistics (covariance) of source and target features.

    L_CORAL = (1/4d^2) * ||C_s - C_t||_F^2
    """
    d = source_features.size(1)

    # Center the features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    cov_source = (source_centered.T @ source_centered) / (source_features.size(0) - 1)
    cov_target = (target_centered.T @ target_centered) / (target_features.size(0) - 1)

    # Frobenius norm of difference
    loss = torch.norm(cov_source - cov_target, p="fro") ** 2 / (4 * d * d)
    return loss


# ──────────────────────────────────────────────────────────────────────
# Strategy 1: Fine-Tuning
# ──────────────────────────────────────────────────────────────────────

class FineTuner:
    """
    Fine-tune a pretrained model on target subject data.

    Two modes:
    - freeze_features=True: Only train the classifier head (feature extraction mode)
    - freeze_features=False: Fine-tune all layers with lower learning rate
    """

    def __init__(self, model: nn.Module, freeze_features: bool = True):
        self.model = model
        self.freeze_features = freeze_features

    def prepare(self, n_classes: int, lr: float, weight_decay: float = 1e-4):
        """
        Prepare model for fine-tuning.

        Args:
            n_classes: Number of target classes (may differ from source)
            lr: Learning rate for fine-tuning
            weight_decay: L2 regularization
        """
        # Replace classifier if class count differs
        if hasattr(self.model, 'classifier'):
            old_classifier = self.model.classifier
            # Find the linear layer
            for module in old_classifier.modules():
                if isinstance(module, nn.Linear):
                    if module.out_features != n_classes:
                        in_features = module.in_features
                        self.model.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features, n_classes),
                        )
                    break

        if self.freeze_features:
            # Freeze feature extractor parameters
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
            # Only optimize classifier
            optimizer = torch.optim.Adam(
                self.model.classifier.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # Discriminative learning rates
            optimizer = torch.optim.Adam([
                {"params": self.model.feature_extractor.parameters(), "lr": lr * 0.1},
                {"params": self.model.classifier.parameters(), "lr": lr},
            ], weight_decay=weight_decay)

        return optimizer


# ──────────────────────────────────────────────────────────────────────
# Strategy 2: Domain Adaptation (DAN / CORAL)
# ──────────────────────────────────────────────────────────────────────

class DomainAdaptationModel(nn.Module):
    """
    Domain Adaptive Neural Network.
    Adds MMD or CORAL loss between source and target feature distributions
    to the classification loss during training.
    """

    def __init__(self, base_model: nn.Module, adaptation_type: str = "mmd"):
        """
        Args:
            base_model: Pretrained model with feature_extractor and classifier
            adaptation_type: 'mmd' or 'coral'
        """
        super().__init__()
        self.feature_extractor = base_model.feature_extractor
        self.classifier = base_model.classifier
        self.adaptation_type = adaptation_type

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with domain adaptation.

        Returns:
            source_logits: Classification logits for source
            domain_loss: MMD or CORAL domain adaptation loss
        """
        source_features = self.feature_extractor(source)
        source_features_flat = source_features.view(source_features.size(0), -1)

        target_features = self.feature_extractor(target)
        target_features_flat = target_features.view(target_features.size(0), -1)

        source_logits = self.classifier(source_features)

        if self.adaptation_type == "mmd":
            domain_loss = compute_mmd(source_features_flat, target_features_flat)
        elif self.adaptation_type == "coral":
            domain_loss = compute_coral(source_features_flat, target_features_flat)
        else:
            domain_loss = torch.tensor(0.0, device=source.device)

        return source_logits, domain_loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference."""
        features = self.feature_extractor(x)
        return self.classifier(features)


# ──────────────────────────────────────────────────────────────────────
# Strategy 3: Progressive Neural Network
# ──────────────────────────────────────────────────────────────────────

class ProgressiveBlock(nn.Module):
    """
    Progressive neural network block with lateral connections.
    Receives features from a frozen source column and learns to combine
    them with new target-specific features.
    """

    def __init__(self, source_block: nn.Module, n_channels_in: int, n_channels_out: int):
        super().__init__()
        # New column block (trainable)
        self.new_block = copy.deepcopy(source_block)
        # Lateral adapter: maps source features to target space
        self.lateral = nn.Sequential(
            nn.Conv2d(n_channels_out, n_channels_out, (1, 1), bias=False),
            nn.BatchNorm2d(n_channels_out),
        )

    def forward(self, x: torch.Tensor, source_features: torch.Tensor) -> torch.Tensor:
        new_features = self.new_block(x)
        lateral_features = self.lateral(source_features)
        return new_features + lateral_features


class ProgressiveNet(nn.Module):
    """
    Progressive Neural Network for BCI Transfer Learning.

    Maintains a frozen source column and adds a new target column with
    lateral connections, allowing knowledge transfer while preserving
    source knowledge.

    Reference: Rusu et al., "Progressive Neural Networks", 2016
    """

    def __init__(self, source_model: nn.Module, n_classes: int):
        super().__init__()
        # Freeze source model entirely
        self.source_model = copy.deepcopy(source_model)
        for param in self.source_model.parameters():
            param.requires_grad = False

        # Build target column with lateral connections
        self.target_model = copy.deepcopy(source_model)

        # Lateral adapters between source and target feature extractor blocks
        self.lateral_scale = nn.Parameter(torch.tensor(0.1))

        # New classifier for target
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(source_model.feature_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Source features (frozen)
        with torch.no_grad():
            source_features = self.source_model.get_features(x)

        # Target features
        target_features = self.target_model.get_features(x)

        # Combine with learnable scaling
        combined = target_features + self.lateral_scale * source_features
        return self.classifier(combined)


# ──────────────────────────────────────────────────────────────────────
# Strategy 4: Multi-Source Transfer
# ──────────────────────────────────────────────────────────────────────

class MultiSourceTransfer(nn.Module):
    """
    Multi-Source Transfer Learning.

    Trains separate models on each source subject and learns to weight
    their contributions for the target subject. Uses attention-based
    weighting to dynamically combine source knowledge.
    """

    def __init__(self, source_models: List[nn.Module], n_classes: int):
        """
        Args:
            source_models: List of pretrained source models (will be frozen)
            n_classes: Number of target classes
        """
        super().__init__()
        self.n_sources = len(source_models)

        # Freeze all source models
        self.source_models = nn.ModuleList()
        for model in source_models:
            frozen = copy.deepcopy(model)
            for param in frozen.parameters():
                param.requires_grad = False
            self.source_models.append(frozen)

        feature_dim = source_models[0].feature_dim

        # Attention mechanism to weight source contributions
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * self.n_sources, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_sources),
        )

        # Combined classifier
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from all source models
        all_features = []
        with torch.no_grad():
            for source_model in self.source_models:
                features = source_model.get_features(x)
                all_features.append(features)

        stacked = torch.stack(all_features, dim=1)  # (B, n_sources, feature_dim)
        concat = torch.cat(all_features, dim=1)  # (B, n_sources * feature_dim)

        # Compute attention weights
        weights = F.softmax(self.attention(concat), dim=1)  # (B, n_sources)
        weights = weights.unsqueeze(-1)  # (B, n_sources, 1)

        # Weighted combination
        combined = (stacked * weights).sum(dim=1)  # (B, feature_dim)

        return self.classifier(combined)


# ──────────────────────────────────────────────────────────────────────
# Strategy 5: Adversarial Domain Adaptation (DANN)
# ──────────────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN).
    Learns domain-invariant features by adversarial training with a
    gradient reversal layer.

    Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", 2016
    """

    def __init__(self, base_model: nn.Module, n_classes: int, alpha: float = 1.0):
        super().__init__()
        self.feature_extractor = copy.deepcopy(base_model.feature_extractor)
        self.alpha = alpha

        feature_dim = base_model.feature_dim

        # Task classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, n_classes),
        )

        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # source=0, target=1
        )

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha is None:
            alpha = self.alpha

        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)

        # Task prediction
        class_logits = self.classifier(features)

        # Domain prediction with gradient reversal
        reversed_features = GradientReversal.apply(features_flat, alpha)
        domain_logits = self.domain_classifier[1:](reversed_features)  # Skip Flatten

        return class_logits, domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)


# ──────────────────────────────────────────────────────────────────────
# Transfer Learning Manager
# ──────────────────────────────────────────────────────────────────────

class TransferLearningManager:
    """
    High-level manager for applying different TL strategies.
    """

    def __init__(self, strategy: str, base_model: nn.Module, n_classes: int, device: str = "cuda"):
        self.strategy = strategy
        self.base_model = base_model
        self.n_classes = n_classes
        self.device = device

    def prepare_model(
        self,
        source_models: Optional[List[nn.Module]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Prepare a model for transfer learning based on strategy.

        Returns:
            model: Configured model ready for target fine-tuning
        """
        if self.strategy == "fine_tune":
            return self._prepare_fine_tune(**kwargs)
        elif self.strategy == "feature_extract":
            return self._prepare_feature_extract(**kwargs)
        elif self.strategy == "domain_adapt":
            return self._prepare_domain_adapt(**kwargs)
        elif self.strategy == "progressive":
            return self._prepare_progressive(**kwargs)
        elif self.strategy == "multi_source":
            return self._prepare_multi_source(source_models, **kwargs)
        elif self.strategy == "dann":
            return self._prepare_dann(**kwargs)
        else:
            raise ValueError(f"Unknown TL strategy: {self.strategy}")

    def _prepare_fine_tune(self, **kwargs) -> nn.Module:
        model = copy.deepcopy(self.base_model)
        # Replace classifier for target classes
        ft = FineTuner(model, freeze_features=kwargs.get("freeze_features", False))
        ft.prepare(self.n_classes, lr=kwargs.get("lr", 1e-4))
        return model

    def _prepare_feature_extract(self, **kwargs) -> nn.Module:
        model = copy.deepcopy(self.base_model)
        ft = FineTuner(model, freeze_features=True)
        ft.prepare(self.n_classes, lr=kwargs.get("lr", 1e-3))
        return model

    def _prepare_domain_adapt(self, **kwargs) -> nn.Module:
        adapt_type = kwargs.get("adaptation_type", "mmd")
        return DomainAdaptationModel(self.base_model, adapt_type)

    def _prepare_progressive(self, **kwargs) -> nn.Module:
        return ProgressiveNet(self.base_model, self.n_classes)

    def _prepare_multi_source(self, source_models: List[nn.Module], **kwargs) -> nn.Module:
        return MultiSourceTransfer(source_models, self.n_classes)

    def _prepare_dann(self, **kwargs) -> nn.Module:
        return DANN(self.base_model, self.n_classes, alpha=kwargs.get("alpha", 1.0))
