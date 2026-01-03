# Dual-Conditional-Inversion
**(NeurIPS 2025) Official Pytorch implementation of the paper "DCI: Dual-Conditional Inversion for Boosting Diffusion-Based Image Editing"**

[Arxiv](https://arxiv.org/abs/2506.02560)

## 🚀 Getting Started

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lzxhh/Dual-Conditional-Inversion.git
cd Dual-Conditional-Inversion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Stable Diffusion Model
   The code will automatically download the Stable Diffusion v1.4 model from HuggingFace on first run. 
   Alternatively, you can specify a local model path using `--model_path` argument.

4. Download Datasets and Benchmarks
   Please follow the instructions in [PnPInversion](https://github.com/cure-lab/PnPInversion) to download the datasets and benchmarks.

   
## Quick Start

### Basic Usage

```bash
python run_dci_P2P.py \
    --input images/0.jpg \
    --source "a round cake with orange frosting on a wooden plate" \
    --target "a square cake with orange frosting on a wooden plate" \
    --blended_word "round square" \
    --output output/
```

### Batch Processing

Configure your batch processing settings following the format of `selected.json`, then run:
```bash
python run_dci_P2P.py --output output/
```

## Usage

### Main Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str | `''` | Input image path |
| `--source` | str | - | Source prompt (describes original image) |
| `--target` | str | - | Target prompt (describes edited image) |
| `--blended_word` | str | - | Word pair for blending (space-separated) |
| `--K_round` | int | `10` | Optimization rounds |
| `--num_of_ddim_steps` | int | `50` | DDIM inference steps |
| `--guidance_scale` | float | `7.5` | Guidance scale |
| `--output` | str | `"dci_test"` | Output directory |
| `--json` | str | `"./data/mapping_file.json"` | JSON file path for batch processing |



## Citation

```
@article{li2025dci,
  title={DCI: Dual-Conditional Inversion for Boosting Diffusion-Based Image Editing},
  author={Li, Zixiang and Wang, Haoyu and Wang, Wei and Tan, Chuangchuang and Wei, Yunchao and Zhao, Yao},
  journal={arXiv preprint arXiv:2506.02560},
  year={2025}
}

```

## Acknowledgments

This code is built on [diffusers](https://github.com/huggingface/diffusers/), [Stable Diffusion](https://github.com/CompVis/stable-diffusion)  and [SPDInv](https://github.com/leeruibin/SPDInv).