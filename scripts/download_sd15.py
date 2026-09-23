#!/usr/bin/env python3
"""Download the Stable Diffusion 1.5 Diffusers checkpoint used by TICoE."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Stable Diffusion 1.5 for TICoE.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPOSITORY)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stable-diffusion-v1-5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
    )
    print(f"[TICoE] Stable Diffusion checkpoint downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
