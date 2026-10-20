"""Attention rollout for the ViT classifier.

Implements the method from Abnar & Zuidema, "Quantifying Attention Flow in
Transformers" (2020): attention weights from every encoder layer are averaged
across heads, combined with the residual (identity) connection, and multiplied
together across layers. The result is a single map showing how much each
output patch attends back to each input patch, which we visualise as a
heatmap over the original canvas — surfacing whether the model is keying off
broad background brushwork or focal subject matter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def attention_rollout(
    model, pixel_values: torch.Tensor, discard_ratio: float = 0.0
) -> np.ndarray:
    """Run a forward pass with attentions enabled and compute the rollout map.

    Args:
        model: a `ViTForImageClassification` (or `.vit` backbone) in eval mode.
        pixel_values: a single preprocessed image tensor, shape (1, 3, H, W).
        discard_ratio: fraction of lowest attention values to zero out per
            layer before rolling out, which can sharpen the resulting map.

    Returns:
        A 2D numpy array (patch_grid x patch_grid) of rollout attention mass
        received by each patch from the [CLS] token, normalised to [0, 1].
    """
    outputs = model(pixel_values=pixel_values, output_attentions=True)
    # outputs.attentions: tuple of (num_layers) tensors, each
    # (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len)

    for attn in attentions:
        attn_heads_avg = attn.mean(dim=1)[0]  # average over heads -> (seq_len, seq_len)

        if discard_ratio > 0:
            flat = attn_heads_avg.flatten()
            n_discard = int(flat.numel() * discard_ratio)
            if n_discard > 0:
                threshold = flat.kthvalue(n_discard).values
                attn_heads_avg = torch.where(
                    attn_heads_avg < threshold, torch.zeros_like(attn_heads_avg), attn_heads_avg
                )

        # Add identity for the residual connection, then re-normalise rows.
        attn_with_residual = attn_heads_avg + torch.eye(seq_len)
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        rollout = attn_with_residual @ rollout

    # Attention flowing from the [CLS] token (index 0) to every patch token.
    cls_to_patches = rollout[0, 1:]
    grid_size = int(np.sqrt(cls_to_patches.numel()))
    rollout_map = cls_to_patches.reshape(grid_size, grid_size).numpy()

    rollout_map = rollout_map - rollout_map.min()
    if rollout_map.max() > 0:
        rollout_map = rollout_map / rollout_map.max()
    return rollout_map


def overlay_rollout_on_image(
    original_image: Image.Image, rollout_map: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Resize the coarse patch-grid rollout map up to the image resolution and
    blend it over the original artwork as a red-toned heatmap overlay.
    """
    import matplotlib.cm as cm

    w, h = original_image.size
    rollout_tensor = torch.tensor(rollout_map).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(rollout_tensor, size=(h, w), mode="bilinear", align_corners=False)
    upsampled = upsampled.squeeze().numpy()

    heatmap = (cm.get_cmap("inferno")(upsampled)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    base = original_image.convert("RGB")
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended


def save_rollout_examples(
    model,
    dataset,
    label_names: list[str],
    device,
    out_dir: str | Path,
    num_examples: int = 12,
) -> None:
    """Convenience batch-runner: sample a handful of test images, compute
    rollout maps, and save side-by-side overlay images for a qualitative report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for idx in range(min(num_examples, len(dataset))):
        pixel_values, label = dataset[idx]
        pixel_values = pixel_values.unsqueeze(0).to(device)

        rollout_map = attention_rollout(model, pixel_values)

        # Undo normalisation for a viewable base image.
        img_np = pixel_values.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)
        base_image = Image.fromarray((img_np * 255).astype(np.uint8))

        overlay = overlay_rollout_on_image(base_image, rollout_map)
        overlay.save(out_dir / f"example_{idx:03d}_{label_names[label]}.png")
