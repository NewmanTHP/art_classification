"""Interactive Gradio demo: upload an artwork, get a predicted style with
confidence scores and an attention-rollout overlay showing where the model
is looking.

Usage:
    python app.py
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data import build_transforms
from src.interpretability import attention_rollout, overlay_rollout_on_image
from src.model import build_vit_model
from src.utils import get_device, load_checkpoint, load_config

CONFIG_PATH = "configs/config.yaml"
CHECKPOINT_PATH = "checkpoints/vit_best.pt"

cfg = load_config(CONFIG_PATH)
device = get_device()

_eval_transform = build_transforms(cfg, cfg["data"]["image_size"], train=False)
_to_pil = transforms.ToPILImage()

_model = None
_label_names: list[str] = []


def _lazy_load_model():
    """Load the trained checkpoint on first request rather than at import time,
    so the file still imports cleanly (e.g. for tests) before training exists.
    """
    global _model, _label_names
    if _model is not None:
        return _model, _label_names

    if not Path(CHECKPOINT_PATH).exists():
        raise gr.Error(
            f"No trained checkpoint found at '{CHECKPOINT_PATH}'. "
            "Run `python -m src.train --config configs/config.yaml` first."
        )

    ckpt = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    _label_names = ckpt["label_names"]
    _model = build_vit_model(
        num_classes=len(_label_names),
        checkpoint=cfg["model"]["vit_checkpoint"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=False,
    ).to(device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    return _model, _label_names


def predict(image):
    if image is None:
        return None, None

    model, label_names = _lazy_load_model()

    pixel_values = _eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()

    top_k = min(5, len(label_names))
    top_probs, top_idx = probs.topk(top_k)
    confidences = {label_names[i]: float(p) for p, i in zip(top_probs, top_idx)}

    rollout_map = attention_rollout(model, pixel_values)
    overlay = overlay_rollout_on_image(image, rollout_map)

    return confidences, overlay


with gr.Blocks(title="WikiArt-ViT: Artistic Style Classifier") as demo:
    gr.Markdown(
        """
        # WikiArt-ViT: Artistic Style Classifier
        Upload a painting to get the predicted artistic movement, plus an
        **attention rollout** overlay showing which regions of the canvas the
        Vision Transformer weighted most heavily for its prediction.
        """
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload artwork")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predicted style")
            attention_output = gr.Image(label="Attention rollout overlay")

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, attention_output])
    image_input.change(fn=predict, inputs=image_input, outputs=[label_output, attention_output])

if __name__ == "__main__":
    demo.launch()
