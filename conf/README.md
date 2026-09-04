# Configs

`conf/config.yaml` is the base Hydra config. The main app is:

```bash
PYTHONPATH=src python -m dresg.apps.train
```

The YAML tree is the single source of runtime values and required-value markers. Hydra composes it with config groups and validates the result against the value-free Structured Config schema in `src/dresg/config.py`; the schema defines names and types without duplicating YAML defaults.

## Public Groups

- `experiment/`: reusable experiment templates and the smoke config.
- `view_selection/`: paper active-view presets for released LLFF/TNT scenes.

Compose a released preset by dataset and scene:

```bash
PYTHONPATH=src python -m dresg.apps.train \
  view_selection=llff/fern \
  data.scene_dir=/path/to/llff/fern \
  data.base_ply=/path/to/point_cloud.ply \
  data.style_image=/path/to/style.png \
  data.output_dir=/path/to/output \
  artifacts.video.path=/path/to/fern_video_path.npz
```

The view preset supplies both `data.views` and selection provenance under `view_selection`.

`artifacts.video.path` alone controls video rendering. A local scene-bound `.npz` created by `tools/path.py` enables video; `null` disables it. Trajectory-generation settings belong to that artifact and are intentionally absent from the training schema.

Hydra writes the resolved task configuration to `<data.output_dir>/.hydra/`. This metadata and the final `point_cloud.ply` allow `tools/render.py` to regenerate train views or video without training again.

The public repository keeps only reusable DReSG configs. Long sweep matrices and paper-specific ablation queues are intentionally excluded. Unknown or removed keys are rejected rather than translated through compatibility aliases.
