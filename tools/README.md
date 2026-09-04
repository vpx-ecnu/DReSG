# Tools

This directory contains thin public CLI entrypoints:

1. `select_views.py`: compute depth-gated per-Gaussian support from a Gaussian PLY and write a reusable Hydra view preset.
2. `path.py`: generate a reusable scene-bound `.npz` camera path for optional video inference.
3. `render.py`: regenerate train-view renders or video from a run's final `point_cloud.ply` and `.hydra/config.yaml`.
4. `evaluate_unified_metrics.py`: evaluate a standard qualitative layout and transactionally write paper-table metrics.

Use `--help` on each command for its complete interface. Camera trajectories are generated only by `path.py`; training never falls back to constructing one. A training config with `artifacts.video.path: null` skips video, while a path value enables it. `render.py video --path ...` can supply or override the path after training.

The metric tool reports only paper-table columns:

- `CLIP-S`
- `DINO-C`
- short/long consistency `LPIPS` and `RMSE`
- training time/memory
- pure render FPS/memory

Metric computation, result-layout validation, and performance-file readers live under `src/dresg/evaluation/`. The tools do not launch training sweeps or external baseline repositories; all external resources are supplied through explicit local paths.
