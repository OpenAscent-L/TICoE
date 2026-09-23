"""Continuous Convex Concept Manifold utilities for TICoE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_prompt_sequence(tokenizer, text_encoder, prompt: str, device: torch.device) -> torch.Tensor:
    """Encode one prompt into a full token-level text embedding."""
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).input_ids.to(device)
    return text_encoder(input_ids)[0]


@torch.no_grad()
def build_cccm_prompt_bank(
    tokenizer,
    text_encoder,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode and normalize all prompt-bank entries."""
    bank = torch.cat(
        [encode_prompt_sequence(tokenizer, text_encoder, prompt, device) for prompt in prompts],
        dim=0,
    )
    return F.layer_norm(bank, bank.shape[-1:])


@torch.no_grad()
def sample_cccm_embedding(
    bank: torch.Tensor,
    temperature: float = 0.7,
    noise_std: float = 0.001,
) -> torch.Tensor:
    """Sample one convex text condition from the CCCM prompt bank."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    prompt_count = bank.size(0)
    concentration = torch.full(
        (prompt_count,),
        1.0 / temperature,
        device=bank.device,
        dtype=bank.dtype,
    )
    weights = torch.distributions.Dirichlet(concentration).sample().view(
        prompt_count,
        1,
        1,
    )
    mixed = (bank * weights).sum(dim=0, keepdim=True)

    if noise_std > 0:
        mixed = mixed + torch.randn_like(mixed) * noise_std

    return F.layer_norm(mixed, mixed.shape[-1:])


def load_ticoe_prompt_bank(path: Path) -> Tuple[str, list[str]]:
    """Load and validate a TICoE prompt-bank JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt bank must be a JSON object: {path}")

    concept = data.get("concept")
    prompts = data.get("prompts")

    if not isinstance(concept, str) or not concept.strip():
        raise ValueError(f"Prompt bank has an invalid concept field: {path}")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt bank has no prompts: {path}")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError(f"Prompt bank contains an invalid prompt: {path}")

    return concept.strip(), [prompt.strip() for prompt in prompts]
