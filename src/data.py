"""WikiArt data loading, class-balance filtering, splitting, and augmentation.

The pipeline is deliberately conservative with augmentation: strong distortions
(large rotations, heavy color shifts, cutout) can destroy the stylistic cues
(palette, brush continuity) that the model is meant to learn, so we stick to
mild crops, subtle color jitter, and horizontal flips.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]  # matches ViT image processor defaults


@dataclass
class DataConfig:
    dataset_name: str = "huggan/wikiart"
    label_column: str = "style"
    min_samples_per_class: int = 300
    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    cache_dir: str = "data/cache"


def build_transforms(cfg: dict[str, Any], image_size: int, train: bool) -> transforms.Compose:
    """Build the torchvision transform pipeline for train vs. eval splits."""
    if train:
        aug = cfg.get("augmentation", {})
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=tuple(aug.get("random_resized_crop_scale", [0.85, 1.0])),
                ),
                transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip_prob", 0.5)),
                transforms.ColorJitter(
                    brightness=aug.get("color_jitter_brightness", 0.1),
                    contrast=aug.get("color_jitter_contrast", 0.1),
                    saturation=aug.get("color_jitter_saturation", 0.1),
                    hue=aug.get("color_jitter_hue", 0.02),
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class WikiArtDataset(Dataset):
    """Thin wrapper around a HuggingFace `datasets` split that applies a torchvision
    transform and remaps raw label ids to a contiguous, filtered label space.
    """

    def __init__(self, hf_split, label_map: dict[int, int], transform: transforms.Compose):
        self.hf_split = hf_split
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        example = self.hf_split[idx]
        image = example["image"].convert("RGB")
        raw_label = example["label"]
        label = self.label_map[raw_label]
        return self.transform(image), label


def load_and_prepare_datasets(
    cfg: dict[str, Any],
) -> tuple[WikiArtDataset, WikiArtDataset, WikiArtDataset, list[str]]:
    """Load WikiArt from the HF Hub, drop rare classes, and produce stratified
    train/val/test torch Datasets plus the ordered list of retained class names.
    """
    from datasets import load_dataset  # local import: heavy dependency, only needed here

    dcfg = cfg["data"]
    raw = load_dataset(dcfg["dataset_name"], split="train", cache_dir=dcfg["cache_dir"])

    label_feature = raw.features["label"]
    all_class_names = label_feature.names

    # 1. Filter rare classes to preserve balanced representational depth.
    counts = Counter(raw["label"])
    keep_ids = sorted(
        cls_id for cls_id, n in counts.items() if n >= dcfg["min_samples_per_class"]
    )
    keep_id_set = set(keep_ids)
    filtered = raw.filter(lambda ex: ex["label"] in keep_id_set)

    # Remap kept raw ids -> contiguous 0..K-1 ids, preserving name order.
    label_map = {raw_id: new_id for new_id, raw_id in enumerate(keep_ids)}
    label_names = [all_class_names[raw_id] for raw_id in keep_ids]

    # 2. Stratified train/val/test split.
    train_frac = dcfg["train_split"]
    val_frac = dcfg["val_split"]
    test_frac = dcfg["test_split"]
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "splits must sum to 1.0"

    split_1 = filtered.train_test_split(
        test_size=(val_frac + test_frac), seed=dcfg["seed"], stratify_by_column="label"
    )
    remainder_test_frac = test_frac / (val_frac + test_frac)
    split_2 = split_1["test"].train_test_split(
        test_size=remainder_test_frac, seed=dcfg["seed"], stratify_by_column="label"
    )

    train_hf, val_hf, test_hf = split_1["train"], split_2["train"], split_2["test"]

    image_size = dcfg["image_size"]
    train_ds = WikiArtDataset(train_hf, label_map, build_transforms(cfg, image_size, train=True))
    val_ds = WikiArtDataset(val_hf, label_map, build_transforms(cfg, image_size, train=False))
    test_ds = WikiArtDataset(test_hf, label_map, build_transforms(cfg, image_size, train=False))

    return train_ds, val_ds, test_ds, label_names
