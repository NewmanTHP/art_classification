"""Evaluation: top-1/top-5 accuracy, macro F1, per-class precision/recall, and
an optional side-by-side benchmark against the ResNet-50 baseline.

Usage:
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/vit_best.pt
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/vit_best.pt --baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader

from src.data import load_and_prepare_datasets
from src.model import build_resnet_baseline, build_vit_model
from src.utils import get_device, load_checkpoint, load_config


@torch.no_grad()
def collect_logits(model, loader, device, is_vit: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(pixel_values=images).logits if is_vit else model(images)
        all_logits.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, label_names: list[str], top_k: int) -> dict:
    preds = logits.argmax(axis=1)
    top1 = float((preds == labels).mean())
    k = min(top_k, logits.shape[1])
    top5 = float(top_k_accuracy_score(labels, logits, k=k, labels=np.arange(logits.shape[1])))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    report = classification_report(
        labels, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds).tolist()
    return {
        "top1_accuracy": top1,
        f"top{k}_accuracy": top5,
        "macro_f1": macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def evaluate(cfg: dict, checkpoint_path: str, include_baseline: bool) -> None:
    device = get_device()
    _train_ds, _val_ds, test_ds, label_names = load_and_prepare_datasets(cfg)
    tcfg = cfg["training"]
    test_loader = DataLoader(
        test_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"]
    )

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    assert ckpt["label_names"] == label_names, (
        "Checkpoint label space does not match the current filtered dataset. "
        "Re-run with the same config used for training."
    )

    vit = build_vit_model(
        num_classes=len(label_names),
        checkpoint=cfg["model"]["vit_checkpoint"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=False,
    ).to(device)
    vit.load_state_dict(ckpt["model_state_dict"])

    ecfg = cfg["evaluation"]
    report_dir = Path(ecfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating ViT on the held-out test split...")
    vit_logits, vit_labels = collect_logits(vit, test_loader, device, is_vit=True)
    results = {"vit": compute_metrics(vit_logits, vit_labels, label_names, ecfg["top_k"])}

    if include_baseline:
        print("Evaluating ResNet-50 baseline for comparison...")
        resnet = build_resnet_baseline(
            num_classes=len(label_names), arch=cfg["model"]["resnet_baseline"]
        ).to(device)
        # NOTE: this evaluates a freshly-initialised-head baseline. For a fair
        # benchmark, train it first with an equivalent loop (see src/train.py
        # as a template) and load its checkpoint here instead.
        resnet_logits, resnet_labels = collect_logits(resnet, test_loader, device, is_vit=False)
        results["resnet_baseline"] = compute_metrics(
            resnet_logits, resnet_labels, label_names, ecfg["top_k"]
        )

    out_path = report_dir / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nViT  — top1: {results['vit']['top1_accuracy']:.4f} | macro F1: {results['vit']['macro_f1']:.4f}")
    if include_baseline:
        rb = results["resnet_baseline"]
        print(f"ResNet-50 — top1: {rb['top1_accuracy']:.4f} | macro F1: {rb['macro_f1']:.4f}")
    print(f"\nFull report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the WikiArt ViT classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--baseline", action="store_true", help="Also evaluate the ResNet-50 baseline."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.baseline)


if __name__ == "__main__":
    main()
