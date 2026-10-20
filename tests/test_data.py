"""Lightweight, network-free smoke tests for the data pipeline building blocks.

Full integration testing against the live WikiArt dataset is intentionally
left out of unit tests (it requires a large download); these tests only
check the pieces that don't need the actual dataset.
"""
from __future__ import annotations

import torch
from PIL import Image

from src.data import build_transforms

BASE_CONFIG = {
    "data": {"image_size": 224},
    "augmentation": {
        "random_resized_crop_scale": [0.85, 1.0],
        "color_jitter_brightness": 0.1,
        "color_jitter_contrast": 0.1,
        "color_jitter_saturation": 0.1,
        "color_jitter_hue": 0.02,
        "horizontal_flip_prob": 0.5,
    },
}


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (300, 250), color=(120, 60, 200))


def test_eval_transform_output_shape():
    transform = build_transforms(BASE_CONFIG, image_size=224, train=False)
    tensor = transform(_dummy_image())
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_train_transform_output_shape_and_range():
    transform = build_transforms(BASE_CONFIG, image_size=224, train=True)
    tensor = transform(_dummy_image())
    assert tensor.shape == (3, 224, 224)
    # Normalised with mean=std=0.5, so values should roughly fall in [-1, 1]
    assert tensor.min() >= -1.01
    assert tensor.max() <= 1.01


def test_train_transform_is_stochastic():
    transform = build_transforms(BASE_CONFIG, image_size=224, train=True)
    img = _dummy_image()
    torch.manual_seed(0)
    out_a = transform(img)
    torch.manual_seed(1)
    out_b = transform(img)
    assert not torch.equal(out_a, out_b)
