"""Checkpoint helpers for TICoE training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_ticoe_checkpoint(
    path: Path,
    step: int,
    unet,
    hvrl,
    optimizer,
    arguments: Dict[str, Any],
) -> None:
    """Save the complete trainable state used by TICoE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "unet": unet.state_dict(),
            "ticoe_hvrl": hvrl.state_dict(),
            "optimizer": optimizer.state_dict(),
            "arguments": arguments,
        },
        path,
    )


def export_ticoe_unet(unet, output_dir: Path) -> None:
    """Export the final edited U-Net in Diffusers format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output_dir, safe_serialization=False)
