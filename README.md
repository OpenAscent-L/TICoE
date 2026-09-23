<div align="center">

# TICoE

### Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-6A5ACD.svg)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.15829-b31b1b.svg)](https://arxiv.org/abs/2604.15829)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Accepted to CVPR 2026**

Jun Li · Lizhi Xiong · Ziqiang Li · Weiwei Jiang · Zhangjie Fu · Yong Li · Guo-Sen Xie

</div>

---

## ✨ Overview

**TICoE** is a text-image collaborative concept-erasure framework for diffusion models. It combines two complementary components:

- **Continuous Convex Concept Manifold (CCCM)**: multiple semantically related prompts are encoded by the Stable Diffusion text encoder and sampled through a Dirichlet-weighted convex combination.
- **Hierarchical Visual Representation Learning (HVRL)**: self-generated reference images are encoded into noisy diffusion latents and fused across the scales `{1.0, 0.75, 0.5}` with Transformer encoder layers.

The trainable U-Net is optimized using a negative classifier-free-guidance target constructed from a frozen original U-Net.

This repository provides the cleaned TICoE training pipeline used for the paper, including reference-image generation, prompt banks, full-state checkpointing, and final edited U-Net export.

<div align="center">
  <a href="https://arxiv.org/abs/2604.15829">
    <img src="https://arxiv.org/html/2604.15829v1/fig/fig_1_1.png" alt="TICoE framework" width="1000"/>
  </a>
  <br>
  <em>Figure 2. Overview of TICoE. The framework constructs a continuous convex concept manifold from multiple prompts and encodes hierarchical visual representations to achieve precise and faithful concept erasure while preserving unrelated content.</em>
</div>

---

## 🧱 Repository Structure

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
├── checkpoints/
│   └── .gitkeep
├── data/
│   └── reference_images/
│       └── .gitkeep
├── outputs/
│   └── .gitkeep
└── ticoe/
    ├── __init__.py
    ├── checkpoint.py
    ├── diffusion.py
    ├── hvrl.py
    └── prompt_bank.py
```

Only files required by the TICoE training pipeline are included. Legacy trainers, obsolete entry points, unused experimental utilities, and unrelated baseline modules are intentionally excluded.

---

## ⚙️ Environment Setup

Clone the repository and create the environment:

```bash
git clone https://github.com/OpenAscent-L/TICoE.git
cd TICoE

conda create -n ticoe python=3.10 -y
conda activate ticoe
pip install -r requirements.txt
```

The released code directly depends on:

```text
torch
torchvision
pillow
transformers
diffusers
huggingface-hub
safetensors
tqdm
```

The paper reports experiments on an **NVIDIA RTX A6000 (48 GB)**. If needed, install the PyTorch build that matches the CUDA driver on your machine.

---

## 📥 Stable Diffusion 1.5 Preparation

The paper uses **Stable Diffusion 1.5** unless otherwise specified.

Download the Diffusers-format checkpoint:

```bash
python scripts/download_sd15.py \
    --output_dir ./checkpoints/stable-diffusion-v1-5
```

Then set:

```bash
export SD15_PATH=./checkpoints/stable-diffusion-v1-5
```

The training code expects the standard Diffusers component structure, including:

```text
scheduler/
text_encoder/
tokenizer/
unet/
vae/
```

A Hugging Face model identifier may also be passed directly through `--ckpt_path`.

---

## 🖼️ Reference-Image Generation

Before TICoE training, the paper constructs a self-generated reference-image set with a clean Stable Diffusion model.

Supplementary Table 4 specifies **200 images per concept** with the following templates:

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

Generate all five concepts:

```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept all
```

The released generator implements the settings explicitly specified in the paper: the five prompt templates above and `n=200`.

The supplementary material additionally mentions filtering low-quality samples with pretrained-classifier scores and optionally appending descriptive phrases when a plain prompt produces poor generations. Because the paper does not specify the classifier, threshold, or descriptive-phrase list, those undocumented choices are not fabricated in this release.

---

## 🧠 Prompt Banks and CCCM

The prompt banks used by TICoE are stored in:

```text
configs/prompt_banks/
```

Each file contains the target concept and its semantically related prompts. Example:

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

For a prompt bank `B = [e_1, ..., e_N]`, TICoE samples a convex textual condition using:

```text
alpha = 1 / tau
w ~ Dirichlet(alpha)
e_c = sum_i w_i e_i
```

Layer normalization is applied to the prompt bank and to the sampled textual condition. The default temperature is `tau = 0.7`.

---

## 🚀 Train TICoE

After generating the 200 reference images, train one concept with:

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```

The command automatically loads:

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

`training_checkpoint.pt` stores the trainable U-Net, TICoE HVRL module, optimizer state, training step, and training arguments. `final_unet/` contains the edited U-Net in Diffusers format.

### Train the five paper concepts

```bash
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept gun
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept nudity
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept tench
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept van_gogh
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept church
```

---

## 🔬 Training Procedure

Each TICoE optimization step follows this pipeline:

1. Randomly sample one image from the self-generated reference set.
2. Encode the image with the frozen VAE.
3. Sample a random diffusion timestep and add DDPM noise.
4. Resize the noisy latent at scales `{1.0, 0.75, 0.5}`.
5. Flatten and concatenate the multi-scale latent tokens.
6. Add sinusoidal positional embeddings.
7. Process the tokens with the TICoE HVRL Transformer encoder layers.
8. Extract the original-resolution tokens and apply residual fusion.
9. Condition the frozen and trainable U-Nets on the sampled CCCM embedding.
10. Construct the negative-CFG target with the frozen U-Net.
11. Jointly optimize the trainable U-Net and TICoE HVRL using the MSE erasure loss.

For efficiency, the conditional and unconditional frozen-U-Net predictions are evaluated in one batched forward pass. This is mathematically equivalent to two separate frozen-U-Net forward passes.

---

## 📐 Hyperparameters

### Settings described in the paper

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

| Argument / setting | Default | Description |
| --- | ---: | --- |
| `--iterations` | `500` | Optimization steps |
| `--image_number` | `200` | Reference images used for training |
| `--num_inference_steps` | `50` | Diffusion timestep bins |
| HVRL learning rate | `5e-5` | `5 x` the U-Net learning rate |
| `--prompt_resample_interval` | `200` | CCCM condition resampling interval |
| `--noise_std` | `0.001` | Optional Gaussian perturbation standard deviation |
| `--save_iter` | `100` | Full-state checkpoint update interval |

The paper describes Gaussian perturbation as optional but does not report a numerical standard deviation. The public code exposes `--noise_std` explicitly. Set `--noise_std 0.0` to disable this perturbation.

---

## 🛠️ Useful Options

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

List all available arguments:

```bash
python TICoE_train.py --help
```

---

## 📊 Evaluation

For attack-based evaluation, we follow the evaluation protocol used by [**UnlearnDiffAtk (ECCV 2024)**](https://github.com/OPTML-Group/Diffusion-MU-Attack), the official implementation of *To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now*.

### Standard Metrics

Following the paper, TICoE is evaluated with the following standard erasure and generation-quality metrics:

- **ASR ↓**: attack success rate for inducing the erased concept.
- **UDA ↓**: adversarial evaluation of whether optimized prompts can recover the erased concept.
- **P4D ↓**: red-teaming attack success rate under adversarially crafted prompts.
- **FID ↓**: distributional image-quality metric evaluated on benign generations.
- **CLIP ↑**: text-image semantic alignment measured with CLIP-ViT-Large-Patch14.

### Our Metric: MCP

In addition to the standard metrics above, we introduce **Morpho-Contextual Concept Preservation (MCP) ↑** to explicitly measure whether concepts that are semantically distinct but morphologically or contextually related to the erased target remain preserved.

For example, when erasing **gun**, MCP evaluates preservation of related but non-target concepts such as **camera**. A higher MCP indicates better contextual fidelity and more precise concept erasure.

The evaluation prompts and concept-specific classifiers follow the settings described in the paper and supplementary material.

---

## 📖 Citation

If you find TICoE useful, please cite:

```bibtex
@InProceedings{Li_2026_CVPR,
    author    = {Li, Jun and Xiong, Lizhi and Li, Ziqiang and Jiang, Weiwei and Fu, Zhangjie and Li, Yong and Xie, Guo-Sen},
    title     = {Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37653-37663}
}
```

---

<div align="center">

**TICoE · CVPR 2026**

</div>
