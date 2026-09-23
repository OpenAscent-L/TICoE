TICoE
Official implementation of TICoE, introduced in Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration.
TICoE combines two components from the paper:
Continuous Convex Concept Manifold (CCCM): a prompt bank is encoded by the Stable Diffusion text encoder and sampled through a Dirichlet-weighted convex combination.
Hierarchical Visual Representation Learning (HVRL): self-generated reference images are encoded into noisy diffusion latents and fused across the scales `{1.0, 0.75, 0.5}` with Transformer encoder layers.
The edited U-Net is optimized with the concept-erasure objective defined by the frozen original U-Net and negative classifier-free guidance.
Quick start
```bash
git clone https://github.com/OpenAscent-L/TICoE.git
cd TICoE

conda create -n ticoe python=3.10 -y
conda activate ticoe
pip install -r requirements.txt

export SD15_PATH=stable-diffusion-v1-5/stable-diffusion-v1-5

python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench

python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```
On first use, Diffusers downloads the Stable Diffusion 1.5 checkpoint into the Hugging Face cache. You may also replace `SD15_PATH` with a local Diffusers-format Stable Diffusion 1.5 directory.
Repository structure
```text
TICoE/
├── TICoE_train.py
├── README.md
├── requirements.txt
├── configs/
│   └── prompt_banks/
│       ├── gun.json
│       ├── nudity.json
│       ├── tench.json
│       ├── van_gogh.json
│       └── church.json
├── data_generation/
│   └── generate_reference_images.py
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
Only files used by the TICoE training pipeline are included. Legacy trainers, old entry points, IP-Adapter code, baseline implementations, example outputs, and unused experimental utilities have been removed.
1. Environment
The package versions below are taken from the provided TICoE project requirements and restricted to packages directly imported by the cleaned release code.
```text
Python 3.10 recommended
torch==2.3.0
torchvision==0.18.0
pillow==10.3.0
transformers==4.49.0
diffusers==0.27.0
tqdm==4.66.4
```
Create the environment:
```bash
conda create -n ticoe python=3.10 -y
conda activate ticoe
pip install -r requirements.txt
```
PyTorch must be installed with a CUDA build compatible with your local NVIDIA driver. If your system requires a platform-specific PyTorch installation command, install the matching CUDA build first and then install the remaining requirements.
The paper reports experiments on an NVIDIA RTX A6000 (48 GB).
2. Stable Diffusion 1.5 preparation
The paper uses Stable Diffusion 1.5 unless otherwise specified. TICoE expects a Diffusers-format checkpoint containing at least:
```text
stable-diffusion-v1-5/
├── scheduler/
├── text_encoder/
├── tokenizer/
├── unet/
└── vae/
```
The code accepts either a local Diffusers-format directory or a Hugging Face model identifier. The current Stable Diffusion 1.5 Diffusers repository can be used directly:
```bash
export SD15_PATH=stable-diffusion-v1-5/stable-diffusion-v1-5
```
`diffusers` downloads and caches the checkpoint automatically on first use. If you already maintain a local copy, set `SD15_PATH` to that directory instead. Both reference-image generation and TICoE training use the same clean checkpoint.
3. Generate the reference images
Before training, the paper generates a self-generated reference-image set with a clean Stable Diffusion model. Table 4 in the supplementary material specifies 200 images per target concept and the following templates:
Concept	Number	Generation template
nudity	200	`a photo of nudity`
Van Gogh	200	`a painting drawn by Van Gogh`
church	200	`a photo of church`
tench	200	`a photo of tench`
gun	200	`a photo of gun`
Generate one concept:
```bash
python data_generation/generate_reference_images.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```
This creates:
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
Reference-image preparation note
The supplementary material additionally states that low-quality samples were filtered using pretrained-classifier scores and that descriptive phrases could be appended for concepts whose plain prompt produced poor images. The paper does not provide the classifier/threshold or the descriptive-phrase list. To avoid fabricating undocumented choices, the released generator implements the fully specified and reproducible part of Table 4: the exact templates above and `n=200` images per concept.
4. Prompt banks
The five prompt banks used by the cleaned release are stored in:
```text
configs/prompt_banks/
```
Each JSON file has the form:
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
During training, TICoE:
encodes every prompt with the Stable Diffusion text encoder;
applies layer normalization to the prompt embeddings;
samples non-negative normalized weights from a symmetric Dirichlet distribution;
constructs a convex combination of the prompt embeddings;
optionally adds Gaussian perturbation and applies layer normalization again.
The Dirichlet concentration follows the paper:
```text
alpha = 1 / tau
```
with the default temperature `tau = 0.7`.
5. Train TICoE
After the 200 reference images have been generated, train a target concept with one command:
```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept tench
```
By default, the trainer reads:
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
The final edited U-Net is stored in `final_unet/`. The complete training checkpoint contains the edited U-Net, TICoE HVRL module, optimizer state, training step, and training arguments.
Example commands for all five concepts
```bash
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept gun
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept nudity
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept tench
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept van_gogh
python TICoE_train.py --ckpt_path "$SD15_PATH" --concept church
```
6. TICoE training pipeline
For each optimization step, the released trainer follows the final TICoE implementation:
sample one reference image from the self-generated concept set;
encode it with the frozen VAE;
sample a random diffusion timestep and add DDPM noise;
resize the noisy latent at scales `{1.0, 0.75, 0.5}`;
flatten and concatenate the multi-scale tokens;
add sinusoidal positional embeddings;
process the tokens with the TICoE HVRL Transformer layers;
extract the original-resolution tokens and apply residual fusion;
condition the frozen and trainable U-Nets on the sampled CCCM embedding;
construct the negative-CFG target with the frozen U-Net;
optimize the trainable U-Net and TICoE HVRL jointly with MSE loss.
The frozen U-Net conditional and unconditional predictions are batched into one forward pass. This is computationally equivalent to two separate frozen-U-Net forward passes.
7. Hyperparameters
Paper-reported settings
The paper and supplementary material explicitly report:
Setting	Value
Backbone	Stable Diffusion 1.5
Reference images	200
Batch size	1
Adam learning rate	`1e-5`
Temperature `tau`	`0.7`
Negative guidance `gamma`	`1.0`
HVRL residual coefficient `lambda`	`0.5`
HVRL scales	`{1.0, 0.75, 0.5}`
Release settings retained from the final training code
The following settings are implementation details retained from the final training code but are not all explicitly specified in the paper text:
Argument	Default	Notes
`--iterations`	`500`	Final training-code default
`--num_inference_steps`	`50`	Final training-code default
HVRL learning rate	`5e-5`	`5 x` the U-Net learning rate, retained from the final code
`--prompt_resample_interval`	`200`	Final training-code behavior
`--noise_std`	`0.001`	Release default for the optional Gaussian perturbation described in the paper
`--save_iter`	`100`	Checkpoint interval
The paper defines Gaussian perturbation as optional but does not report a numerical noise standard deviation. The public release therefore exposes `--noise_std` explicitly. Set `--noise_std 0.0` to disable the perturbation.
8. Common options
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
Change the GPU:
```bash
python TICoE_train.py \
    --ckpt_path "$SD15_PATH" \
    --concept gun \
    --device cuda:1
```
Inspect all arguments:
```bash
python TICoE_train.py --help
```
9. Evaluation scope
The paper evaluates TICoE with ASR, UDA, P4D, FID, CLIP, and MCP. The provided clean release focuses on the TICoE training pipeline and does not fabricate evaluation implementations that were not present in the supplied project code.
10. Citation
```bibtex
@article{li2026beyond,
  title={Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration},
  author={Li, Jun and Xiong, Lizhi and Li, Ziqiang and Jiang, Weiwei and Fu, Zhangjie and Li, Yong and Xie, Guo-Sen},
  journal={arXiv preprint arXiv:2604.15829},
  year={2026}
}
```
