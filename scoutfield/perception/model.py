"""
The classifier: EfficientNet-B0 with an MC-dropout-capable head.

Why EfficientNet-B0
-------------------
It is the smallest member of a family with strong ImageNet transfer, it fine-tunes
in minutes on a free-tier GPU, and it exports cleanly to ONNX for the deployment
item on the roadmap. The project's binding constraint is a free Kaggle GPU with a
weekly quota, so a backbone that needs a day of training is out of scope
regardless of its accuracy.

Why the head keeps dropout as a real layer
------------------------------------------
Roadmap item 7 compares three uncertainty methods: temperature scaling,
MC-dropout, and deep ensembles. MC-dropout requires dropout to stay *active at
inference*, which means it must be an ``nn.Dropout`` module that can be switched
back on after ``model.eval()`` — not a functional call baked into forward, and not
something the export path silently removes.

Binary output, one logit
------------------------
The environment is binary and the pilot's instrument acts on a single scalar
logit with the decision boundary at 0. Matching that exactly — one output unit,
BCEWithLogits, threshold at 0 — keeps temperature scaling in this project
identical in form to temperature scaling in the pilot, which is what makes the
two phases comparable.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

_BACKBONES = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
}


class BinaryClassifier(nn.Module):
    """Pretrained backbone with a dropout head emitting one raw logit.

    No sigmoid anywhere in ``forward``. The calibration code scales the *logit*,
    and a sigmoid buried in the model is the most likely cause of the
    accuracy-invariance invariant failing.
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 head_dropout: float = 0.3):
        super().__init__()
        if backbone not in _BACKBONES:
            raise KeyError(f"unsupported backbone '{backbone}'; known: {sorted(_BACKBONES)}")
        factory, weights, n_features = _BACKBONES[backbone]
        self.backbone = factory(weights=weights if pretrained else None)
        # Dropout stays a real nn.Dropout module so MC-dropout can switch it back
        # on after .eval() — a functional call could not be re-enabled, and the
        # ONNX export path would silently drop it.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit per image, shape ``(batch,)``."""
        return self.backbone(x).squeeze(-1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the feature extractor, leaving the head trainable.

        Toggled rather than rebuilt so the optimizer's parameter groups and the
        checkpoint's state_dict stay valid across the freeze boundary.
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = trainable


def build_model(config):
    """Construct the backbone and head from ``config['model']``."""
    model_cfg = config.section("model")
    return BinaryClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        head_dropout=float(model_cfg.get("head_dropout", 0.3)),
    )


def enable_mc_dropout(model) -> None:
    """Eval mode everywhere except dropout, which stays stochastic.

    Asserts that at least one dropout layer ended up active: silently sampling a
    deterministic model would produce zero epistemic uncertainty and read as a
    finding rather than a bug.
    """
    model.eval()
    active = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            active += 1
    if active == 0:
        raise RuntimeError("no nn.Dropout layers found; MC-dropout would be a no-op")
