<div align="center">

<h1>DReSG</h1>
<h3>Diffusion Residuals for Stylized Gaussian Splatting</h3>

<p>
  <a href="https://github.com/lzlcs">Zhongliang Liu</a> ·
  <a href="https://github.com/Grav1tum">Wenjie Liu</a> ·
  <a href="https://ihpdep.github.io/">Yang Li</a>
</p>
<p>East China Normal University</p>
<p><strong>Pacific Graphics 2026 · Conference Track</strong></p>

<p>
  <a href="https://vpx-ecnu.github.io/DReSG-website/"><img src="https://img.shields.io/badge/Project-Page-4c8bf5?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2608.29048"><img src="https://img.shields.io/badge/arXiv-2608.29048-b31b1b?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv"></a>
  <a href="https://vpx-ecnu.github.io/DReSG-website/static/pdfs/DReSG.pdf"><img src="https://img.shields.io/badge/Paper-PDF-2f6f9f?style=flat-square&amp;logo=adobeacrobatreader&amp;logoColor=white" alt="Paper PDF"></a>
</p>

</div>

<p align="center">
  <img src="https://vpx-ecnu.github.io/DReSG-website/static/images/project/teaser.webp" width="100%" alt="DReSG reference-guided 3D Gaussian stylization teaser">
</p>
<p align="center"><em>Reference-guided stylization across multiple scenes and styles.</em></p>

## Overview

DReSG is a research codebase for diffusion-guided 3D Gaussian scene stylization. It renders a small active set of scene views, obtains attention-guided diffusion residual feedback from a reference style image, and writes the result into a shared 3D Gaussian scene so the stylization remains attached to 3D primitives across views.

This release contains the DReSG pipeline, retained DINO-patch regularization, rendering/export helpers, and unified metrics utilities.

## Installation

```bash
git clone https://github.com/vpx-ecnu/DReSG.git
cd DReSG

conda create -n dresg python=3.10
conda activate dresg

# Install a CUDA-compatible PyTorch build for your machine first.
# Example only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

The release was tested with Python 3.10.14, PyTorch 2.3.0, CUDA 12.1, and gsplat 1.5.3. See:

- [Installation and model weights](docs/INSTALL.md)
- [Data and Gaussian scene inputs](docs/DATA.md)
- [Usage](docs/USAGE.md)

## Workflow

Prepare a COLMAP-style scene, its matching Fast-PGSR/Graphdeco-compatible `point_cloud.ply`, and a style image. For a new scene, select active views first:

```bash
PYTHONPATH=src python tools/select_views.py \
  --scene-dir /path/to/llff/fern \
  --base-ply /path/to/point_cloud.ply \
  --output-dir outputs/view_selection/fern \
  --dataset llff \
  --scene fern \
  --factor 4
```

Generate a scene-bound video path when video output is wanted:

```bash
PYTHONPATH=src python tools/path.py llff \
  --scene-dir /path/to/llff/fern \
  --factor 4 \
  --camera-source all \
  --n-frames 120 \
  --coord-mode flip_yz \
  --radius-scale 1.0 \
  --output /path/to/fern_video_path.npz
```

Then run DReSG. Released scenes can use the exact active-view presets under `conf/view_selection/`:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
python -m dresg.apps.train \
  view_selection=llff/fern \
  data.scene_dir=/path/to/llff/fern \
  data.base_ply=/path/to/point_cloud.ply \
  data.style_image=/path/to/style.png \
  data.output_dir=outputs/fern_style \
  artifacts.video.path=/path/to/fern_video_path.npz \
  runtime.device=cuda:0
```

`artifacts.video.path` is the video switch: omit it or leave it `null` to skip video rendering. A successful run always writes `point_cloud.ply`. The output directory must be new; Hydra records the composed run under `.hydra/`, and DReSG rejects other pre-existing artifacts.

Saved runs can be rendered again without retraining:

```bash
PYTHONPATH=src python tools/render.py train-views \
  --run-dir outputs/fern_style --device cuda:0

PYTHONPATH=src python tools/render.py video \
  --run-dir outputs/fern_style --device cuda:0
```

The video command uses the path saved in `.hydra/config.yaml`; pass `--path` to supply or override it. See [Usage](docs/USAGE.md) for the complete LLFF/TNT workflow and evaluation command.

## Quick Checks

```bash
PYTHONPATH=src python -m dresg.apps.train --cfg job
PYTHONPATH=src python -m dresg.apps.train experiment=smoke_test --cfg job
PYTHONPATH=src python -m pytest tests/unit -q
```

## Citation

Until the archival proceedings metadata become available, please cite the arXiv preprint:

```bibtex
@misc{liu2026dresg,
  title         = {DReSG: Diffusion Residuals for Stylized Gaussian Splatting},
  author        = {Liu, Zhongliang and Liu, Wenjie and Li, Yang},
  year          = {2026},
  eprint        = {2608.29048},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.29048},
  note          = {Accepted to Pacific Graphics 2026 (Conference Track)}
}
```

The repository also provides [CITATION.cff](CITATION.cff). It will be updated with the final proceedings DOI and bibliographic information after publication.

## Responsible Use

The research pipeline disables the Stable Diffusion safety checker because it analyzes and optimizes internal diffusion features. Do not expose the unchecked pipeline as a public generation service. Review generated outputs and comply with model licenses and acceptable-use terms.

## License

DReSG is released under the [Gaussian-Splatting License](LICENSE.md) for non-commercial research and evaluation use.
