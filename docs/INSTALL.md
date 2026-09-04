# Installation

DReSG assumes Python 3.10+ and a CUDA-enabled PyTorch build matched to your driver. The release was tested with Python 3.10.14, PyTorch 2.3.0, CUDA 12.1, and gsplat 1.5.3.

## Core Environment

```bash
conda create -n dresg python=3.10
conda activate dresg

# Choose the correct command from https://pytorch.org/get-started/locally/
# Example only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

DReSG uses [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian rasterization. Install it following its official instructions if a compatible wheel is unavailable, then check:

```bash
python -c "import gsplat; print('gsplat ok')"
```

The repository does not vendor third-party runtime repositories. Installed packages remain governed by their own licenses and terms.

## Stable Diffusion And DINO

The default configuration refers to these external model weights, which DReSG does not redistribute:

- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5): `stable-diffusion-v1-5/stable-diffusion-v1-5`
- [DINOv2 base](https://huggingface.co/facebook/dinov2-base): `facebook/dinov2-base`

Review each current model card, license, and acceptable-use terms before downloading or using the weights. Online runs obtain them through Hugging Face when they are not already cached. This behavior is enabled by default:

```bash
runtime.offline_models=false
```

For offline execution, prepare the weights independently in a Hugging Face cache or local directories, prohibit downloads, and optionally point the model fields at those directories:

```bash
runtime.offline_models=true \
guidance.backbone.model_id=/path/to/local/stable-diffusion-v1-5 \
image_loss.content3d_dino_model=/path/to/local/dinov2-base
```

When `runtime.offline_models=true`, both the diffusion model and DINO use local files only and fail if a required file is unavailable. DINO is loaded only when `image_loss.lambda_content3d` is positive.

The research runtime disables the Stable Diffusion safety checker while optimizing internal features. Do not expose it as an unchecked public generation service. Review generated output and comply with the model's license and acceptable-use restrictions.

Optional evaluation additionally uses explicitly selected CLIP, DINO, LPIPS, and RAFT weights. `tools/evaluate_unified_metrics.py --offline-models` prohibits downloads and requires every requested model to be cached locally.

## Running From A Checkout

Run commands from the repository root so Hydra can load `conf/`, and keep `src/` on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Smoke Checks

```bash
PYTHONPATH=src python -m dresg.apps.train --cfg job
PYTHONPATH=src python -m dresg.apps.train experiment=smoke_test --cfg job
PYTHONPATH=src python -m pytest tests/unit -q
```

The composition commands inspect configuration without loading datasets or model weights. See [DATA.md](DATA.md) before attempting a training run.
