#!/usr/bin/env python3
"""Generate the self-generated reference-image set used by TICoE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm


PROMPT_TEMPLATES = {
    "nudity": "a photo of nudity",
    "van_gogh": "a painting drawn by Van Gogh",
    "church": "a photo of church",
    "tench": "a photo of tench",
    "gun": "a photo of gun",
}


def generate_concept_images(
    pipeline,
    concept: str,
    output_root: Path,
    num_images: int,
) -> None:
    """Generate the requested number of reference images for one concept."""
    prompt = PROMPT_TEMPLATES[concept]
    concept_dir = output_root / concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    for index in tqdm(range(num_images), desc=f"Generating {concept}"):
        image = pipeline(
            prompt=prompt,
            height=512,
            width=512,
        ).images[0]
        image.save(concept_dir / f"{concept}_{index:04d}.png")


def parse_args() -> argparse.Namespace:
    """Parse reference-image generation arguments."""
    parser = argparse.ArgumentParser(description="Generate TICoE reference images.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Stable Diffusion 1.5 checkpoint in Diffusers format or a model identifier.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        choices=("gun", "nudity", "tench", "van_gogh", "church", "all"),
    )
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--output_root", type=str, default="./data/reference_images")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    return args


def main() -> None:
    """Generate TICoE reference images with the clean diffusion model."""
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.ckpt_path,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
    ).to(device)

    output_root = Path(args.output_root).expanduser().resolve()
    concepts = tuple(PROMPT_TEMPLATES) if args.concept == "all" else (args.concept,)

    for concept in concepts:
        generate_concept_images(
            pipeline=pipeline,
            concept=concept,
            output_root=output_root,
            num_images=args.num_images,
        )


if __name__ == "__main__":
    main()
