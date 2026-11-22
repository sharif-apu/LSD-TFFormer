<div align="center">

# LSD: Low-light Smartphone Dataset

### Illuminating Darkness: Learning to Enhance Low-light Images In-the-Wild
**WACV 2026**

[**Paper**](https://arxiv.org/abs/2503.06898) · [**Dataset**](https://huggingface.co/datasets/ARM4588/Lowlight-Smartphone-Dataset) · [**Results**](#results) · [**Talk**](https://www.youtube.com/watch?v=3GQWxt8QlwE) · [**Poster**](assets/poster_wacv.pdf)

[![arXiv](https://img.shields.io/badge/arXiv-2503.06898-b31b1b.svg)](https://arxiv.org/abs/2503.06898)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LSD-yellow.svg)](https://huggingface.co/datasets/ARM4588/Lowlight-Smartphone-Dataset)
[![Data license](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Code license](https://img.shields.io/badge/code-MIT-blue.svg)](#license)

<img src="assets/lsd_teaser.jpg" width="90%" alt="LSD pairs: low-light smartphone input next to its aligned ground truth, outdoor and indoor">

</div>

**LSD** is the largest real-world paired dataset for single-shot low-light image enhancement: **6,425 aligned low/normal-light pairs** captured in the wild on **15 smartphone sensors**, from **0.1 to 200 lux**, plus **LSD-U**, **2,117 unpaired images** from unseen devices for testing generalization. This repository also contains **TFFormer**, our luminance–chrominance decoupled enhancement model.

## News
- **[Coming Soom]** We are launching LSD benchmarking. Details coming soon.
- **[Coming Soon]** We are organizing the **NTIRE 2027** low-light enhancement challenge on LSD at CVPR 2027. Details coming soon.
- **[Nov 2026]** TFFormer checkpoint released and the LSD-U unpaired test set fixed.
- **[Jan 2026]** LSD was the official dataset of the **NTIRE 2026 Single-Shot Low-Light Enhancement Challenge** at CVPR 2026: [Track 1 (in-the-wild)](https://www.codabench.org/competitions/11864/), [Track 2 (joint denoising)](https://www.codabench.org/competitions/12994/), [NTIRE 2026](https://cvlai.net/ntire/2026/).
- **[Jan 2026]** Dataset mirrored on [Hugging Face](https://huggingface.co/datasets/ARM4588/Lowlight-Smartphone-Dataset).

## The dataset

| | LSD (paired) | LSD-U (unpaired) |
|---|---|---|
| Images | 6,425 pairs (6,025 train · 400 test) | 2,117 |
| Sources | 15 smartphone sensors | Video frames 1,217 · DSLR 300 · Android 300 · iPhone 250 · social media 50 |
| Illuminance | 0.1–200 lux, ISO 50–3200 | mixed |
| Resolution | 4K+ full scenes, ~180K training patches | mixed |
| Use | training and full-reference benchmarking | generalization and perceptual evaluation |

Paired data comes in two variants:

- **DLL** (denoised low-light, 3,175 pairs): for low-light enhancement.
- **NLL** (noisy low-light, 3,250 pairs): for joint denoising and enhancement.

**Compared with existing paired datasets**

| Dataset | Sensors | Resolution | Full scene | Pairs |
|---|:-:|:-:|:-:|--:|
| LOL-V1 | 1 | 400×600 | ✗ | 500 |
| LOL-V2 | 1 | 400×600 | ✗ | 789 |
| LSRW | 2 | 960×720 | ✗ | 5,650 |
| NCLLIE | 1 | 4K+ | ✓ | 225 |
| **LSD** | **15** | **4K+** | ✓ | **6,425** |

### Download

| Source | Contents |
|---|---|
| [Hugging Face](https://huggingface.co/datasets/ARM4588/Lowlight-Smartphone-Dataset) | Training patches: `patch_DLL_{input,gt}Patch.tar.gz`, `patch_NLL_{input,gt}Patch.tar.gz` (166 GB total) |
| [Google Drive](https://drive.google.com/drive/folders/1QPT6vbRlA-47N0xNElmVYYR6hx9hGZiw?usp=sharing) | Full scenes, 400-pair test set, LSD-U |

```
LSD/
├── training/
│   ├── full_scenes/
│   └── patches/
│       ├── inputPatchDLL/   gtPatchDLL/
│       └── inputPatchNLL/   gtPatchNLL/
└── testing/
```

## Benchmark

An open LSD benchmark is coming: submit your enhanced results and get scored on hidden ground truth with the same evaluation script for everyone. Until it launches, see the [results](#results) from the paper below.

## TFFormer

### Installation

Tested with Python 3.10, PyTorch 2.5 and CUDA 12 on Linux. The pinned `nvidia-*` and `triton` packages are Linux/CUDA only.

```bash
git clone https://github.com/sharif-apu/LSD-TFFormer.git
cd LSD-TFFormer
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

### Configuration

Set your paths in `mainModule/config.json`:

```json
"gtPath": "./lsd/gtPatch/",
"targetPath": "./lsd/inputPatch/",
"testImagePath": "./lsd_test/",
"checkpointPath": "./modelLog/LowLightEnhancement/checkpoint_TFFormer/"
```

`testImagePath` should contain the `gtPatch` and `inputPatch` folders of the test set.

### Pretrained checkpoint

Download from [Google Drive](https://drive.google.com/drive/folders/1Y9wOSCVOTao2QoyyKu2lXw83bXVzr3Vk?usp=sharing) and point `checkpointPath` to it.

### Training

```bash
python main.py -ts   # train from scratch
python main.py -tr   # resume or fine-tune from checkpointPath
```

Checkpoints and enhanced images are written to `modelLog/`.

> **Fine-tuning note:** the number of training steps/epochs in the config must exceed those stored in the checkpoint, otherwise training stops immediately.

### Results

<p align="center"><img src="assets/lsd_results.png" width="95%" alt="Quantitative comparison on LSD"></p>

Best and second-best results are shown in red and blue.

## Citation

```bibtex
@inproceedings{sharif2026illuminating,
  title={Illuminating Darkness: Learning to Enhance Low-light Images In-the-Wild},
  author={Sharif, SMA and Rehman, Abdur and Abidin, Zain Ul and Dharejo, Fayaz Ali and Timofte, Radu and Naqvi, Rizwan Ali},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={2263--2272},
  year={2026}
}
```

## License

- **Code:** MIT.
- **Dataset (LSD and LSD-U):** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for non-commercial research with attribution. For commercial use, contact us.

## About

LSD took over two years to collect and is self-funded research. We release the data and code openly because we believe research should be accessible to everyone. If LSD or TFFormer helps your work, please cite the paper.

**Authors:** S M A Sharif¹, Abdur Rehman¹, Zain Ul Abidin², Fayaz Ali Dharejo³, Radu Timofte³, Rizwan Ali Naqvi²
¹Opt-AI Inc. · ²Sejong University · ³University of Würzburg

**Contact:** apuism@gmail.com · abdurehman458@gmail.com

**Acknowledgment:** This code builds on [BJDD](https://github.com/sharif-apu/BJDD_CVPR21) and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).
