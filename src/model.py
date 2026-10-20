"""Model builders for the ViT classifier and the ResNet-50 baseline."""
from __future__ import annotations

import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification


def build_vit_model(
    num_classes: int,
    checkpoint: str = "google/vit-base-patch16-224-in21k",
    dropout: float = 0.1,
    freeze_backbone: bool = True,
) -> ViTForImageClassification:
    """Load a pre-trained ViT and swap in a fresh classification head sized to
    `num_classes`. Each 224x224 input is patchified into 16x16 non-overlapping
    patches (196 patches + 1 [CLS] token) and processed by a multi-head
    self-attention encoder with a global receptive field from the first layer.
    """
    config = ViTConfig.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        hidden_dropout_prob=dropout,
    )
    model = ViTForImageClassification.from_pretrained(
        checkpoint, config=config, ignore_mismatched_sizes=True
    )

    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)

    return model


def set_backbone_trainable(model: ViTForImageClassification, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head.

    Used to implement the two-phase transfer-learning schedule: phase 1 trains
    only `model.classifier` with the backbone frozen, phase 2 unfreezes
    everything for low-LR fine-tuning.
    """
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def build_resnet_baseline(num_classes: int, arch: str = "resnet50") -> nn.Module:
    """ImageNet-pretrained CNN baseline for benchmarking against the ViT."""
    import torchvision.models as tv_models

    weights_enum = {
        "resnet50": tv_models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet34": tv_models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet18": tv_models.ResNet18_Weights.IMAGENET1K_V1,
    }
    if arch not in weights_enum:
        raise ValueError(f"Unsupported baseline arch '{arch}'. Choose from {list(weights_enum)}.")

    model_fn = getattr(tv_models, arch)
    model = model_fn(weights=weights_enum[arch])
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
