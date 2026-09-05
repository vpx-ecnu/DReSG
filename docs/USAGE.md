# Running DReSG

Run all commands from the repository root with `PYTHONPATH=src`. Prepare the Gaussian PLY and scene layout described in [DATA.md](DATA.md) before training.

## 1. Select Active Views (Optional)

Skip this step when using a released preset under `conf/view_selection/`, such as `view_selection=llff/fern` or `view_selection=tnt/m60`.

For a new scene, run the paper's depth-gated per-Gaussian selection rule first:

```bash
PYTHONPATH=src python tools/select_views.py \
  --scene-dir /path/to/scene \
  --base-ply /path/to/point_cloud.ply \
  --output-dir outputs/view_selection/my_scene \
  --dataset llff \
  --scene my_scene \
  --device cuda:0 \
  --factor 4
```

The command atomically writes `coverage_selection.json` and `view_selection.yaml`. It stores sparse visible-Gaussian support per view in memory and does not write a dense `views × Gaussians` tensor. Copy the generated preset into a private Hydra config group or pass its `data.views` list to training.

## 2. Generate A Video Path (Optional)

Skip this step when video output is not needed and leave `artifacts.video.path` as `null`. Video trajectories are independent, scene-bound artifacts; training never creates or substitutes one.

For LLFF:

```bash
PYTHONPATH=src python tools/path.py llff \
  --scene-dir /path/to/llff/scene \
  --factor 4 \
  --camera-source all \
  --n-frames 120 \
  --coord-mode flip_yz \
  --radius-scale 1.0 \
  --output /path/to/scene_video_path.npz
```

For TNT:

```bash
PYTHONPATH=src python tools/path.py tnt \
  --scene-dir /path/to/tnt/scene \
  --factor 1 \
  --camera-source all \
  --n-frames 240 \
  --ellipse-scale 1.1 \
  --output /path/to/scene_video_path.npz
```

The artifact stores finite `float32` poses, common source calibration, exact generation metadata, and a reconstruction fingerprint. A path from a different COLMAP reconstruction is rejected before Gaussian or diffusion model loading.

## 3. Train One Scene

LLFF example using a released active-view preset:

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

TNT example:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
python -m dresg.apps.train \
  experiment=tnt \
  view_selection=tnt/m60 \
  data.scene_dir=/path/to/tnt/m60 \
  data.base_ply=/path/to/point_cloud.ply \
  data.style_image=/path/to/style.png \
  data.output_dir=outputs/m60_style \
  artifacts.video.path=/path/to/m60_video_path.npz \
  runtime.device=cuda:0
```

`artifacts.video.path` is the only video control. Leave it `null` or omit the override to skip final video rendering; provide a valid path to render automatically when training completes. `artifacts.save_train_views` and `artifacts.save_style_image` independently control those two exports.

`data.output_dir` must be new. Hydra creates `.hydra/` for the current invocation before the application starts; preflight permits those three metadata files but rejects every other pre-existing artifact.

## Outputs

Every successful run writes:

```text
.hydra/config.yaml
.hydra/hydra.yaml
.hydra/overrides.yaml
summary.json
aggregate_metrics.json
point_cloud.ply
```

Depending on artifact settings, it also writes:

```text
style_image.png
renders/000000.png
renders/000001.png
...
view_manifest.csv
export_metrics.json
video.mp4
```

`point_cloud.ply` is always the final stylized Gaussian scene in canonical binary little-endian SH0 format. `summary.json` is atomically replaced throughout training; `aggregate_metrics.json` appears only after successful completion. `view_manifest.csv` records the exact input-view index for every render and is authoritative during evaluation.

## Render A Saved Run Again

A completed run can regenerate exports from its final `point_cloud.ply` and strict saved `.hydra/config.yaml` without retraining.

Regenerate all input-view renders:

```bash
PYTHONPATH=src python tools/render.py train-views \
  --run-dir outputs/fern_style \
  --device cuda:0
```

Regenerate video using the path saved by the original run:

```bash
PYTHONPATH=src python tools/render.py video \
  --run-dir outputs/fern_style \
  --device cuda:0
```

Supply a path when the saved config has `artifacts.video.path: null`, or override the saved path:

```bash
PYTHONPATH=src python tools/render.py video \
  --run-dir outputs/fern_style \
  --path /path/to/another_scene_bound_path.npz \
  --device cuda:0
```

Train-view bundles and videos are replaced atomically. These commands do not modify training `summary.json` or `aggregate_metrics.json`.

## Inspect Configuration

```bash
PYTHONPATH=src python -m dresg.apps.train --cfg job
PYTHONPATH=src python -m dresg.apps.train experiment=tnt --cfg job
```

Hydra composes YAML values against a strict Structured Config schema. Unknown or removed keys fail composition; no compatibility aliases are provided.

## Metrics

The generic metrics script expects:

```text
qualitative/{method}/{scene}/{style}/renders/*.png
qualitative/{method}/{scene}/{style}/view_manifest.csv
qualitative/{method}/{scene}/{style}/aggregate_metrics.json
qualitative/{method}/{scene}/{style}/export_metrics.json
```

It writes a new output directory containing `paper_metrics.csv`, `paper_metrics.json`, and `paper_metrics_config.json`. Evaluation is fail-fast: missing artifacts, models, or metrics produce no partial result bundle. Pass `--offline-models` to prohibit model downloads; the requested CLIP, DINO, LPIPS AlexNet, and RAFT weights must already be cached.

The default quality protocol matches the original evaluation scripts:

- **CLIP-S:** mean cosine similarity to the style image using CLIP ViT-B/32 and its standard preprocessing.
- **DINO-C:** mean aligned-view cosine similarity using DINOv2-base (ViT-B/14), its pooled output, and the model's image processor (short edge 256, center crop 224 for the original checkpoint).
- **ST/LT consistency:** RAFT-Large `C_T_SKHT_V2` estimates forward/backward flow from the original input photos, not Gaussian base renders or stylized images. By default, all input-camera renders are ordered by their manifest view IDs; ST uses gap 1 and LT uses `floor(N/2)`, each with up to six uniformly spaced, rounded start indices.
- **RMSE:** computed over RGB channels of valid pixels only. Validity requires strictly interior sampling coordinates and forward/backward agreement: squared cycle error `< 0.01 ×` summed squared flow magnitudes `+ 0.5`.
- **LPIPS:** invalid warped pixels are replaced with the corresponding reference-frame pixels before full-image AlexNet LPIPS (`spatial=False`). This is not a masked average of a spatial LPIPS map.

A sampled pair with no valid pixels, non-finite flow or scores, or invalid CLIP/DINO features aborts evaluation rather than contributing a zero or NaN. No pairs are silently dropped. For historical reproduction, use the same images, view manifest, batch size (8 in the original runs), and model checkpoints; the original DINOv2-base snapshot was `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, which can be supplied as a local directory through `--dino-model`.

```bash
PYTHONPATH=src python tools/evaluate_unified_metrics.py \
  --qual-root outputs/my_eval/qualitative \
  --scene-dir /path/to/scene \
  --scene fortress \
  --style 085 \
  --style-path /path/to/styles/085.png \
  --methods dresg \
  --factor 4 \
  --result-dir dresg=outputs/fortress_style \
  --output-dir outputs/my_eval/metrics
```
