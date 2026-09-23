# TICoE

Official implementation of **TICoE**, introduced in **Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration**.

TICoE contains two core components:

- **Continuous Convex Concept Manifold (CCCM):** multiple semantically related prompts are encoded by the Stable Diffusion text encoder and sampled through a Dirichlet-weighted convex combination.
- **Hierarchical Visual Representation Learning (HVRL):** self-generated reference images are encoded into noisy diffusion latents and fused across the scales `{1.0, 0.75, 0.5}` with Transformer encoder layers.

The trainable U-Net is optimized with the concept-erasure objective constructed from the frozen original U-Net and negative classifier-free guidance.

## 1. Repository structure

```text
TICoE/
├── TICoE_train.py
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── prompt_banks/
│       ├── gun.json
│       ├── nudity.json
│       ├── tench.json
│       ├── van_gogh.json
│       └── church.json
├── data_generation/
│   └── generate_reference_images.py
├── scripts/
│   └── download_sd15.py
├── data/
│   └── reference_images/
├── outputs/
└── ticoe/
    ├── __init__.py
    ├── checkpoint.py
    ├── diffusion.py
    ├── hvrl.py
    └── prompt_bank.py
```

Only files required by the TICoE training pipeline are included. Legacy entry points, obsolete trainers, IP-Adapter modules, baseline code, and unused experiment utilities are intentionally excluded.

## 2. Environment setup

The release uses only packages required by the cleaned TICoE code and checkpoint-download workflow.

```bash
git clone https://github.com/OpenAscent-L/TICoE.git
cd TICoE

conda create -n ticoe python=3.10 -y
conda activate ticoe
pip install -r requirements.txt
```

The release keeps the PyTorch, TorchVision, Pillow, Transformers, and tqdm versions used by the supplied project environment. `diffusers` is set to a compatible release for a clean installation with Transformers 4.49 and the current Hugging Face Hub API.

The paper reports experiments on an NVIDIA RTX A6000 (48 GB). For GPU training, install a PyTorch build compatible with the CUDA driver on your machine if the default pip installation is not appropriate for your system.

## 3. Download Stable Diffusion 1.5

The paper uses **Stable Diffusion 1.5** unless otherwise specified. Download the Diffusers-format checkpoint with:

```bash
python scripts/download_sd15.py \
    --output_dir ./checkpoints/stable-diffusion-v1-5
```

Set the checkpoint path:

```bash
export SD15_PATH=./checkpoints/stable-diffusion-v1-5
```

The training code expects the usual Diffusers component structure, including `scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, and `vae/`.

You may also pass a Hugging Face model identifier directly as `--ckpt_path`; Diffusers will download and cache it automatically.

## 4. Generate the 200 reference images

Before TICoE training, the paper uses a clean Stable Diffusion model to generate a self-generated reference-image set for each target concept. Supplementary Table 4 specifies **200 images per concept** and the following prompt templates:

| Concept | Number | Generation prompt |
| --- | ---: | --- |
| nudity | 200 | `a photo of nudity` |
| Van Gogh | 200 | `a painting drawn by Van Gogh` |
| church | 200 | `a photo of church` |
| tench | 200 | `a photo of tench` |
| gun | 200 | `a photo of gun` |

Generate one concept:

```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```

The default output is:

```text
data/reference_images/tench/
├── tench_0000.png
├── tench_0001.png
├── ...
└── tench_0199.png
```

Generate all five paper concepts:

```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept all
```

The released generator follows the explicitly specified Table 4 settings: the concept-specific prompt templates above and `n=200`. The supplementary material additionally mentions filtering low-quality generations using pretrained-classifier scores and optionally appending descriptive phrases when a plain prompt performs poorly, but it does not specify the classifier, threshold, or phrase list. Those undocumented choices are not fabricated in this release.

## 5. Prompt-bank preparation

The five prompt banks used by TICoE are provided under:

```text
configs/prompt_banks/
```

Each file contains the concept name and its semantically related prompts. For example:

```json
{
  "concept": "tench",
  "prompts": [
    "tench",
    "common tench",
    "European tench"
  ]
}
```

TICoE encodes every prompt using the Stable Diffusion text encoder, applies layer normalization, and builds a convex textual condition using symmetric Dirichlet sampling:

```text
alpha = 1 / tau
```

The default temperature is `tau = 0.7`.

## 6. Train TICoE

After generating the reference images, start training with:

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```

The command automatically reads:

```text
configs/prompt_banks/tench.json
data/reference_images/tench/
```

and writes:

```text
outputs/tench/
├── training_checkpoint.pt
└── final_unet/
    ├── config.json
    └── diffusion_pytorch_model.bin
```

`training_checkpoint.pt` stores the trainable U-Net, TICoE HVRL module, optimizer state, training step, and training arguments. `final_unet/` is the edited U-Net exported in Diffusers format.

### Train all five concepts

```bash
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept gun
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept nudity
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept tench
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept van_gogh
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept church
```

## 7. Training procedure

Each optimization step follows the TICoE implementation:

1. Randomly sample one image from the self-generated reference set.
2. Encode the image with the frozen VAE.
3. Sample a random diffusion timestep and add DDPM noise.
4. Resize the noisy latent at scales `{1.0, 0.75, 0.5}`.
5. Flatten and concatenate the multi-scale latent tokens.
6. Add sinusoidal positional embeddings.
7. Process the tokens with the TICoE HVRL Transformer encoder layers.
8. Extract the original-resolution tokens and apply residual fusion.
9. Condition the frozen and trainable U-Nets on the sampled CCCM embedding.
10. Construct the negative-CFG target using the frozen U-Net.
11. Jointly optimize the trainable U-Net and TICoE HVRL with the MSE erasure loss.

The conditional and unconditional frozen-U-Net predictions are batched into one forward pass. This is computationally equivalent to two separate frozen-U-Net forward passes.

## 8. Hyperparameters

### Settings explicitly described in the paper

| Setting | Value |
| --- | ---: |
| Backbone | Stable Diffusion 1.5 |
| Reference images | 200 |
| Batch size | 1 |
| Adam U-Net learning rate | `1e-5` |
| Temperature `tau` | `0.7` |
| Negative guidance `gamma` | `1.0` |
| HVRL residual coefficient `lambda` | `0.5` |
| HVRL scales | `{1.0, 0.75, 0.5}` |

### Final-code implementation settings

The following defaults are retained from the final TICoE implementation or exposed explicitly for the public code:

| Argument | Default | Description |
| --- | ---: | --- |
| `--iterations` | `500` | Optimization steps |
| `--image_number` | `200` | Reference images used for training |
| `--num_inference_steps` | `50` | Diffusion timestep bins |
| HVRL learning rate | `5e-5` | `5 x` the U-Net learning rate |
| `--prompt_resample_interval` | `200` | CCCM condition resampling interval |
| `--noise_std` | `0.001` | Optional Gaussian perturbation standard deviation |
| `--save_iter` | `100` | Full-state checkpoint update interval |

The paper describes Gaussian perturbation as optional but does not report a numerical standard deviation. The public release therefore exposes `--noise_std` explicitly; use `--noise_std 0.0` to disable it.

## 9. Useful options

Use a custom reference-image directory:

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept gun \
    --image_dir /path/to/reference/images
```

Use a custom prompt bank:

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept gun \
    --prompt_bank /path/to/custom_prompt_bank.json
```

Select another GPU:

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept gun \
    --device cuda:1
```

Inspect all available arguments:

```bash
python TICoE_train.py --help
```

## 10. Evaluation scope

The paper evaluates TICoE using **ASR, UDA, P4D, FID, CLIP, and MCP**. This clean release focuses on the TICoE training pipeline and does not add evaluation implementations that were not present in the supplied project code.

## 11. Citation

```bibtex
@article{li2026beyond,
  title={Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration},
  author={Li, Jun and Xiong, Lizhi and Li, Ziqiang and Jiang, Weiwei and Fu, Zhangjie and Li, Yong and Xie, Guo-Sen},
  journal={arXiv preprint arXiv:2604.15829},
  year={2026}
}
```
