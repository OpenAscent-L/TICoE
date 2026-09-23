"""Diffusion-model helpers used by TICoE training."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_ticoe_components(checkpoint: str, device: torch.device):
    """Load the frozen teacher and trainable Stable Diffusion 1.5 components."""
    frozen_unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")
    trainable_unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")

    frozen_unet.requires_grad_(False).eval().to(device)
    trainable_unet.requires_grad_(True).train().to(device)
    vae.requires_grad_(False).eval().to(device)
    text_encoder.requires_grad_(False).eval().to(device)

    move_scheduler_tensors(scheduler, device)
    return frozen_unet, trainable_unet, vae, tokenizer, text_encoder, scheduler


def move_scheduler_tensors(scheduler, device: torch.device) -> None:
    """Move scheduler tensors to the training device."""
    for attribute in ("betas", "alphas", "alphas_cumprod", "one", "timesteps"):
        value = getattr(scheduler, attribute, None)
        if torch.is_tensor(value):
            setattr(scheduler, attribute, value.to(device))


def predict_ticoe_noise(
    latent: torch.Tensor,
    timestep: torch.Tensor,
    unet,
    text_embedding: torch.Tensor,
) -> torch.Tensor:
    """Predict diffusion noise with one TICoE U-Net forward pass."""
    return unet(
        latent.to(unet.device),
        timestep.to(unet.device),
        encoder_hidden_states=text_embedding.to(unet.device),
        return_dict=False,
    )[0]
