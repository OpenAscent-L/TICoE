#!/usr/bin/env python3
"""Train TICoE for one target concept."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ticoe.checkpoint import export_ticoe_unet, save_ticoe_checkpoint
from ticoe.diffusion import load_ticoe_components, move_scheduler_tensors, predict_ticoe_noise
from ticoe.hvrl import TICoEHVRL
from ticoe.prompt_bank import build_cccm_prompt_bank, load_ticoe_prompt_bank, sample_cccm_embedding


REFERENCE_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


class TICoEReferenceImageDataset(Dataset):
    """Reference-image dataset used by TICoE training."""

    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = REFERENCE_IMAGE_TRANSFORM(image.convert("RGB"))
        return tensor, str(path)


def resolve_prompt_bank(concept: str, prompt_bank: str) -> Path:
    """Resolve the prompt bank for the requested concept."""
    if prompt_bank:
        path = Path(prompt_bank).expanduser().resolve()
    else:
        path = PROJECT_ROOT / "configs" / "prompt_banks" / f"{concept}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt bank not found: {path}")
    return path


def resolve_reference_image_dir(concept: str, image_dir: str) -> Path:
    """Resolve the reference-image directory for the requested concept."""
    if image_dir:
        path = Path(image_dir).expanduser().resolve()
    else:
        path = PROJECT_ROOT / "data" / "reference_images" / concept
    if not path.is_dir():
        raise FileNotFoundError(
            f"Reference-image directory not found: {path}. "
            "Generate the images first with data_generation/generate_reference_images.py."
        )
    return path


def collect_reference_images(image_dir: Path, image_number: int) -> list[Path]:
    """Select the requested number of reference images."""
    candidates = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(candidates) < image_number:
        raise RuntimeError(
            f"TICoE requires {image_number} reference images, but only "
            f"{len(candidates)} were found in {image_dir}."
        )
    return random.sample(candidates, image_number)


def train_ticoe(args: argparse.Namespace) -> None:
    """Run TICoE concept-erasure training."""
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    prompt_bank_path = resolve_prompt_bank(args.concept, args.prompt_bank)
    concept_name, prompts = load_ticoe_prompt_bank(prompt_bank_path)

    if not args.prompt_bank and concept_name != args.concept:
        raise ValueError(
            f"Built-in prompt bank concept mismatch: requested {args.concept!r}, "
            f"found {concept_name!r}."
        )

    reference_image_dir = resolve_reference_image_dir(concept_name, args.image_dir)

    frozen_unet, unet, vae, tokenizer, text_encoder, noise_scheduler = load_ticoe_components(
        args.ckpt_path,
        device,
    )

    ticoe_hvrl = TICoEHVRL(
        in_channels=4,
        num_heads=4,
        depth=2,
        scales=(1.0, 0.75, 0.5),
        residual_weight=args.lambda_hvrl,
    ).to(device)
    ticoe_hvrl.train()

    optimizer = torch.optim.Adam(
        [
            {"params": list(unet.parameters()), "lr": args.lr},
            {"params": list(ticoe_hvrl.parameters()), "lr": args.lr * 5.0},
        ]
    )
    criterion = nn.MSELoss()

    noise_scheduler.set_timesteps(args.num_inference_steps)
    move_scheduler_tensors(noise_scheduler, device)

    prompt_bank = build_cccm_prompt_bank(tokenizer, text_encoder, prompts, device)
    unconditional_ids = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).input_ids.to(device)
    with torch.no_grad():
        unconditional_embedding = text_encoder(unconditional_ids)[0]

    image_paths = collect_reference_images(reference_image_dir, args.image_number)
    dataset = TICoEReferenceImageDataset(image_paths)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.iterations)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        )
    loader = DataLoader(**loader_kwargs)

    output_dir = Path(args.output_dir).expanduser().resolve() / concept_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_checkpoint.pt"

    print(f"[TICoE] Concept: {concept_name}")
    print(f"[TICoE] Prompt bank: {prompt_bank_path} ({len(prompts)} prompts)")
    print(f"[TICoE] Reference images: {len(image_paths)} from {reference_image_dir}")
    print(f"[TICoE] Output directory: {output_dir}")

    text_embedding = None

    for step, (image_cpu, _) in enumerate(
        tqdm(loader, desc=f"TICoE training: {concept_name}", total=args.iterations),
        start=1,
    ):
        optimizer.zero_grad(set_to_none=True)

        if (step - 1) % args.prompt_resample_interval == 0:
            text_embedding = sample_cccm_embedding(
                prompt_bank,
                temperature=args.temperature,
                noise_std=args.noise_std,
            ).to(device)

        image_tensor = image_cpu.to(device, non_blocking=True)

        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215

        timestep_bin = random.randrange(args.num_inference_steps)
        low = int(timestep_bin * 1000 / args.num_inference_steps)
        high = int((timestep_bin + 1) * 1000 / args.num_inference_steps)
        timestep = torch.randint(low, high, (1,), device=device, dtype=torch.long)

        noise = torch.randn_like(latent)
        noisy_latent = noise_scheduler.add_noise(latent, noise, timestep)
        fused_latent = ticoe_hvrl(noisy_latent)

        with torch.no_grad():
            teacher_latent = torch.cat([fused_latent, fused_latent], dim=0)
            teacher_timestep = torch.cat([timestep, timestep], dim=0)
            teacher_embedding = torch.cat([text_embedding, unconditional_embedding], dim=0)
            teacher_prediction = predict_ticoe_noise(
                teacher_latent,
                teacher_timestep,
                frozen_unet,
                teacher_embedding,
            )
            conditional_teacher, unconditional_teacher = teacher_prediction.chunk(2, dim=0)

        student_prediction = predict_ticoe_noise(
            fused_latent,
            timestep,
            unet,
            text_embedding,
        )

        target_prediction = unconditional_teacher - args.negative_guidance * (
            conditional_teacher - unconditional_teacher
        )
        loss = criterion(student_prediction, target_prediction)
        loss.backward()
        optimizer.step()

        if args.log_every > 0 and step % args.log_every == 0:
            tqdm.write(f"[TICoE] step={step:04d} loss={loss.item():.6f}")

        if args.save_iter > 0 and step % args.save_iter == 0:
            save_ticoe_checkpoint(
                checkpoint_path,
                step,
                unet,
                ticoe_hvrl,
                optimizer,
                vars(args),
            )

    save_ticoe_checkpoint(
        checkpoint_path,
        args.iterations,
        unet,
        ticoe_hvrl,
        optimizer,
        vars(args),
    )
    export_ticoe_unet(unet, output_dir / "final_unet")
    print(f"[TICoE] Training complete. Final UNet: {output_dir / 'final_unet'}")


def parse_args() -> argparse.Namespace:
    """Parse TICoE training arguments."""
    parser = argparse.ArgumentParser(description="Train TICoE for concept erasure.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Stable Diffusion 1.5 checkpoint in Diffusers format or a model identifier.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="gun",
        choices=("gun", "nudity", "tench", "van_gogh", "church"),
    )
    parser.add_argument(
        "--prompt_bank",
        type=str,
        default="",
        help="Optional custom prompt-bank JSON. Overrides the built-in prompt bank.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Reference-image directory. Defaults to data/reference_images/<concept>.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--image_number", type=int, default=200)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lambda_hvrl", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.001)
    parser.add_argument("--prompt_resample_interval", type=int, default=200)
    parser.add_argument("--save_iter", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.image_number <= 0:
        parser.error("--image_number must be positive")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.lambda_hvrl < 0:
        parser.error("--lambda_hvrl must be non-negative")
    if args.noise_std < 0:
        parser.error("--noise_std must be non-negative")
    if args.prompt_resample_interval <= 0:
        parser.error("--prompt_resample_interval must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch_factor must be positive")

    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_ticoe(parse_args())
