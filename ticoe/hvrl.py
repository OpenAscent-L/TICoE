"""Hierarchical Visual Representation Learning for TICoE."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TICoEHVRL(nn.Module):
    """Multi-scale latent fusion module used by TICoE."""

    def __init__(
        self,
        in_channels: int = 4,
        num_heads: int = 4,
        depth: int = 2,
        scales: Sequence[float] = (1.0, 0.75, 0.5),
        residual_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.residual_weight = float(residual_weight)
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=in_channels,
                    nhead=num_heads,
                    dim_feedforward=in_channels * 4,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latent.shape
        token_groups = []

        for scale in self.scales:
            scaled_size = (int(height * scale), int(width * scale))
            scaled_latent = F.interpolate(
                latent,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )
            token_groups.append(scaled_latent.flatten(2).transpose(1, 2))

        tokens = torch.cat(token_groups, dim=1)
        tokens = tokens + self._sinusoidal_position_embedding(
            sequence_length=tokens.size(1),
            embedding_dim=tokens.size(-1),
            device=tokens.device,
            dtype=tokens.dtype,
        )

        for block in self.transformer_blocks:
            tokens = block(tokens)

        fused_tokens = tokens[:, : height * width, :]
        fused_latent = fused_tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        return latent + self.residual_weight * fused_latent

    @staticmethod
    def _sinusoidal_position_embedding(
        sequence_length: int,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        embedding = torch.zeros(
            sequence_length,
            embedding_dim,
            device=device,
            dtype=torch.float32,
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding.unsqueeze(0).to(dtype=dtype)
