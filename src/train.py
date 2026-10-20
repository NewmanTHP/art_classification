"""Two-phase transfer-learning training loop for the WikiArt ViT classifier.

Phase 1 ("head warm-up"): the pre-trained ViT backbone is frozen and only the
    custom linear classification head is optimised. This lets the head adapt
    to the new label space quickly without disturbing pre-trained features.
Phase 2 ("fine-tuning"): the entire network is unfrozen and trained at a low
    learning rate with a cosine annealing schedule.

Usage:
    python -m src.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import load_and_prepare_datasets
from src.model import build_vit_model, set_backbone_trainable
from src.utils import count_trainable_params, get_device, load_config, save_checkpoint, set_seed


def run_epoch(
    model, loader, device, optimizer=None, scaler: GradScaler | None = None
) -> tuple[float, float]:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            with autocast(enabled=scaler is not None):
                outputs = model(pixel_values=images).logits
                loss = loss_fn(outputs, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(cfg: dict) -> None:
    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, _test_ds, label_names = load_and_prepare_datasets(cfg)
    print(f"Retained {len(label_names)} style classes after rare-class filtering.")
    print(f"Train/val sizes: {len(train_ds)} / {len(val_ds)}")

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"]
    )

    model = build_vit_model(
        num_classes=len(label_names),
        checkpoint=cfg["model"]["vit_checkpoint"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=True,
    ).to(device)

    use_amp = tcfg["fp16"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    checkpoint_dir = Path(tcfg["checkpoint_dir"])
    best_val_acc = 0.0

    # ---- Phase 1: frozen backbone, train the classification head ----
    print(f"\n[Phase 1] Head warm-up — trainable params: {count_trainable_params(model):,}")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg["lr_head"]
    )
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0)

    for epoch in range(tcfg["epochs_head"]):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        scheduler.step()
        print(
            f"  epoch {epoch + 1}/{tcfg['epochs_head']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.1f}s)"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, label_names, checkpoint_dir / "vit_best.pt")

    # ---- Phase 2: unfreeze everything, fine-tune with cosine annealing ----
    set_backbone_trainable(model, trainable=True)
    print(f"\n[Phase 2] Fine-tuning — trainable params: {count_trainable_params(model):,}")
    optimizer = AdamW(
        model.parameters(), lr=tcfg["lr_finetune"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs_finetune"])

    for epoch in range(tcfg["epochs_finetune"]):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        scheduler.step()
        print(
            f"  epoch {epoch + 1}/{tcfg['epochs_finetune']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s)"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, label_names, checkpoint_dir / "vit_best.pt")

    save_checkpoint(model, label_names, checkpoint_dir / "vit_final.pt")
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints written to {checkpoint_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WikiArt ViT classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
