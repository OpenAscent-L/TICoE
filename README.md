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

<div align="center">
  <img src="https://arxiv.org/html/2604.15829v1/fig/fig_1_1.png" alt="TICoE framework" width="1000"/>
</div>

---

## 📌 Overview

**TICoE** is a text-image collaborative concept-erasure framework for diffusion models. It combines:

- **Continuous Convex Concept Manifold (CCCM):** samples continuous textual conditions from multiple semantically related prompts.
- **Hierarchical Visual Representation Learning (HVRL):** fuses self-generated visual representations across multiple latent scales.

The trainable U-Net is optimized against a negative classifier-free-guidance target produced by a frozen original U-Net.

## ⚙️ Installation

```bash
git clone https://github.com/OpenAscent-L/TICoE.git
cd TICoE

conda create -n ticoe python=3.10 -y
conda activate ticoe
pip install -r requirements.txt
```

## 📦 Download model weights

TICoE uses **Stable Diffusion 1.5** by default. Download a Diffusers-format checkpoint into `checkpoints/`:

```bash
python scripts/download_sd15.py \
    --output_dir ./checkpoints/stable-diffusion-v1-5
```

Set the checkpoint path:

```bash
export SD15_PATH=./checkpoints/stable-diffusion-v1-5
```

You may also pass a Hugging Face model identifier directly through `--ckpt_path`.

## 🚀 Run

### 1. Generate reference images

For convenient reproduction, directly generate 200 reference images for each target concept using its corresponding concept-specific prompt template. The released prompt banks cover `gun`, `nudity`, `tench`, `van_gogh`, and `church`.

Generate one concept:

```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```

Generate all concepts:

```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept all
```

Images are saved under `data/reference_images/<concept>/`.

### 2. Train TICoE

```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```

For **Van Gogh**, use `--concept van_gogh`. Results are written to:

```text
outputs/<concept>/
├── training_checkpoint.pt
└── final_unet/
    ├── config.json
    └── diffusion_pytorch_model.bin
```

### 3. Prompt banks and default settings

Prompt banks are stored in `configs/prompt_banks/`. CCCM samples a convex textual condition with:

```text
alpha = 1 / tau
w ~ Dirichlet(alpha)
e_c = sum_i w_i e_i
```

Core defaults:

| Setting | Value |
|---|---:|
| Backbone | Stable Diffusion 1.5 |
| Reference images | 200 |
| Training iterations | 500 |
| U-Net learning rate | `1e-5` |
| HVRL learning rate | `5e-5` |
| CCCM temperature `tau` | `0.7` |
| Negative guidance `gamma` | `1.0` |
| HVRL residual coefficient `lambda` | `0.5` |
| HVRL scales | `{1.0, 0.75, 0.5}` |

Use `python TICoE_train.py --help` for custom image directories, prompt banks, and device selection.

## 📁 Repository Structure

```text
TICoE/
├── TICoE_train.py
├── README.md
├── requirements.txt
├── .gitignore
├── configs/prompt_banks/
│   ├── church.json
│   ├── gun.json
│   ├── nudity.json
│   ├── tench.json
│   └── van_gogh.json
├── data_generation/generate_reference_images.py
├── scripts/download_sd15.py
├── checkpoints/.gitkeep
├── data/reference_images/.gitkeep
├── outputs/.gitkeep
└── ticoe/
    ├── checkpoint.py
    ├── diffusion.py
    ├── hvrl.py
    └── prompt_bank.py
```

## 📊 Evaluation

For attack-based evaluation, we follow the protocol of [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack). The standard evaluation includes ASR, UDA, P4D, FID, and CLIP. **MCP** measures preservation of semantically distinct but morphologically or contextually related concepts.

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
