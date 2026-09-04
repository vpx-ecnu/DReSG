# Data And Gaussian Scene Inputs

DReSG does not require a fixed global data root. Supply explicit local paths for a COLMAP-style scene, its matching Gaussian PLY, and a reference style image.

## Resource Provenance And Rights

The paper experiments use scenes from:

- [LLFF](https://github.com/Fyusion/LLFF)
- [Tanks and Temples](https://www.tanksandtemples.org/)

Reference styles were selected from:

- the [ABC-GS style archive](https://drive.google.com/file/d/10EPUQpH0PE8Mnoxxs1URePtjQZElt--s/view?usp=sharing), linked by the [ABC-GS repository](https://github.com/vpx-ecnu/ABC-GS)
- the [`data/style` collection in InstantStyle-Plus](https://github.com/instantX-research/InstantStyle-Plus/tree/main/data/style)

The Gaussian inputs were produced with Graphdeco-compatible reconstruction workflows related to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [FastGS](https://github.com/fastgs/FastGS), and [PGSR](https://github.com/zju3dv/PGSR). These external projects and generated scenes remain governed by their own licenses and terms.

DReSG redistributes none of these datasets, styles, reconstruction implementations, or generated Gaussian scenes. The links above establish provenance; they do not grant redistribution rights. Obtain resources independently, review the applicable upstream terms, and use only scene and style data you are authorized to use. Do not include third-party style images in a release or example bundle unless their terms permit it.

## Gaussian PLY Format

Fast-PGSR and Graphdeco-style pipelines normally write:

```text
/path/to/reconstruction/point_cloud/iteration_30000/point_cloud.ply
```

Pass that file directly as `data.base_ply`; no conversion step or DReSG-specific Gaussian format is used.

DReSG consumes only these scalar vertex properties:

```text
x y z
f_dc_0 f_dc_1 f_dc_2
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

ASCII and binary PLY files, either byte order, and ordinary numeric input precisions are accepted and converted to `float32`. Normals, higher-order SH coefficients, additional vertex properties, and additional PLY elements are ignored. A deterministic finite mask removes rows with a non-finite value in a consumed property and emits a warning; a file with no finite retained Gaussian is rejected. Every retained quaternion must have non-zero norm.

A successful run always writes `point_cloud.ply`. Output is canonical binary little-endian SH0 PLY with `float32` properties and zero normals. Output tensors are required to have exact shapes, finite values, and non-zero quaternions at the serialization boundary.

The PLY must belong to the same reconstruction and image set as `data.scene_dir`. DReSG intentionally provides no alternate Gaussian input, filename fallback, or migration layer.

## Required Configuration

- `data.scene_dir`: LLFF/TNT/COLMAP-style scene directory.
- `data.base_ply`: matching Fast-PGSR/Graphdeco-compatible `point_cloud.ply`.
- `data.style_image`: reference style image.
- `data.output_dir`: new output directory, preferably outside the Git repository.
- `data.views`: active training view indices.
- `artifacts.video.path`: optional scene-bound `.npz` created by `tools/path.py`. A path value enables final video rendering; `null` disables it.

All active views must share one image resolution. Video export additionally requires a common intrinsic matrix because one K is used along the generated path. These conditions and a provided video-path artifact are checked before Gaussian or diffusion model loading.

The released LLFF reconstructions use COLMAP `SIMPLE_RADIAL` cameras, while Gaussian training and rasterization use pinhole intrinsics. DReSG follows the same approximation as the supplied PLY. Do not mix a Gaussian scene with a differently undistorted image set.

Every scene must contain `sparse/0` and `images/`; a factor greater than one additionally requires `images_<factor>`. DReSG resolves pyramid images automatically. It first matches each registered COLMAP image by relative stem, allowing extension changes. If those names are not preserved, it accepts an LLFF-style pyramid only when `images/` exactly matches the registered COLMAP names, the image counts agree, and the pyramid is named `image000`, `image001`, and so on; that verified case is paired in lexical order. Any ambiguous or incomplete mapping, non-positive factor, duplicate view name or stem, or inconsistent per-camera resolution fails before model loading. Intrinsics are rescaled from the COLMAP camera to the realized pyramid dimensions.

LLFF and built-in spiral trajectories additionally require a finite `poses_bounds.npy` with one row per canonical view and positive `near < far` bounds. Missing trajectory metadata is an error; DReSG does not substitute synthetic bounds or read hidden metadata overrides.

## Suggested Layout

```text
/data/dresg/
  scenes/
    llff/
      fern/
      flower/
    tnt/
      m60/
      train/
  gaussians/
    llff/fern/point_cloud.ply
    llff/flower/point_cloud.ply
    tnt/m60/point_cloud.ply
  styles/
    001.png
    002.png
  paths/
    fern_video_path.npz
```

## Active-View Presets

The released configs contain the exact active views used with the paper Gaussian bases:

| Dataset | Scene | Selected / Candidates | Coverage |
| --- | --- | ---: | ---: |
| LLFF | fern | 17 / 20 | 0.9974 |
| LLFF | flower | 23 / 34 | 0.9889 |
| LLFF | fortress | 18 / 42 | 0.9902 |
| LLFF | horns | 29 / 62 | 0.9815 |
| LLFF | leaves | 24 / 26 | 0.9977 |
| LLFF | orchids | 25 / 25 | 1.0000 |
| LLFF | trex | 21 / 55 | 0.9830 |
| TNT | family | 77 / 152 | 0.9615 |
| TNT | m60 | 60 / 313 | 0.9290 |
| TNT | playground | 55 / 307 | 0.9333 |
| TNT | train | 66 / 301 | 0.9043 |
| TNT | truck | 67 / 251 | 0.9279 |

Select one with `view_selection=<dataset>/<scene>`. For a new scene, generate a preset before training:

```bash
PYTHONPATH=src python tools/select_views.py \
  --scene-dir /path/to/scene \
  --base-ply /path/to/point_cloud.ply \
  --output-dir outputs/view_selection \
  --dataset llff \
  --scene my_scene \
  --device cuda:0 \
  --factor 4
```

The paper selection rule uses depth-gated per-Gaussian maximum support with:

```yaml
target_fraction_of_max: 0.98
min_weight: 1.0e-4
min_marginal_gain_ratio: 0.001
stop_coverage_ratio: 0.9999
max_select: null
```

The greedy process stops when the best remaining view improves average normalized target coverage by less than `0.001`. `room` and `horse` were excluded from the paper-facing scene set, so no final-paper preset is published for them. The command writes `coverage_selection.json` and `view_selection.yaml`; use the generated `data.views` list in training or copy the preset into a private Hydra config group.
