from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dresg.data.cameras import CameraView
from dresg.models.gs import build_gaussian_scene as _build_gaussian_scene
from dresg.models.gs.serialization.ply import load_gaussian_ply, save_gaussian_ply
from dresg.models.gs.serialization.sh import rgb_to_sh0


def _minimal_ply(tmp_path: Path, sh0_values: torch.Tensor | None = None) -> Path:
    count = 10
    sh0 = sh0_values if sh0_values is not None else torch.randn(count, 1, 3)
    path = tmp_path / "point_cloud.ply"
    save_gaussian_ply(
        path,
        splats={
            "means": torch.randn(count, 3),
            "quats": torch.randn(count, 4),
            "scales": torch.log(torch.rand(count, 3).clamp_min(1e-6)),
            "opacities": torch.randn(count),
            "sh0": sh0,
        },
    )
    return path


def build_gaussian_scene(
    ply_path: Path,
    device: torch.device,
    **overrides,
):
    options = {
        "optimize_geometry": False,
        "optimize_quats": False,
        "max_mean_delta": 0.05,
        "max_scale_delta": 0.05,
        "max_quat_delta": 0.02,
    }
    options.update(overrides)
    return _build_gaussian_scene(
        ply_path=ply_path,
        device=device,
        **options,
    )


class TestGaussianScene:
    def test_builder_exposes_one_trainable_param(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cpu"))
        params = scene.appearance_parameters()
        assert len(params) == 1
        assert params[0].shape == (10, 3)
        assert params[0].requires_grad
        assert len(scene.geometry_parameters()) == 0
        assert len(list(scene.buffers())) > 0

    def test_direct_rgb_carrier_optimizes_bounded_rgb(self, tmp_path: Path) -> None:
        sh0 = torch.zeros(10, 1, 3)
        path = _minimal_ply(tmp_path, sh0_values=sh0)
        scene = build_gaussian_scene(path, device=torch.device("cpu"))
        params = scene.appearance_parameters()
        assert len(params) == 1
        assert params[0].shape == (10, 3)
        with torch.no_grad():
            scene.appearance_rgb.fill_(2.0)
        scene.apply_parameter_constraints()
        colors = scene.colors()
        assert colors.shape == (10, 3)
        assert torch.all(colors >= 0.0)
        assert torch.all(colors <= 1.0)

    def test_direct_rgb_ply_saves_degree_zero_colors(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cpu"))
        out_path = tmp_path / "styled.ply"

        scene.save_ply(out_path)

        splats = load_gaussian_ply(out_path)
        expected_sh0 = rgb_to_sh0(scene.colors()).unsqueeze(1).detach().cpu()
        torch.testing.assert_close(splats["sh0"], expected_sh0, rtol=0, atol=0)

    def test_geometry_optimization_uses_bounded_deltas(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(
            path,
            device=torch.device("cpu"),
            optimize_geometry=True,
            optimize_quats=True,
            max_mean_delta=0.02,
            max_scale_delta=0.03,
            max_quat_delta=0.04,
        )
        assert len(scene.geometry_mean_scale_parameters()) == 2
        assert len(scene.geometry_quat_parameters()) == 1
        with torch.no_grad():
            for param in scene.geometry_parameters():
                param.fill_(100.0)
        mean_delta = scene.means() - scene.base_means
        scale_delta = scene.scales_log() - scene.base_scales_log
        assert mean_delta.abs().max() <= 0.020001
        assert scale_delta.abs().max() <= 0.030001
        assert scene.quat_delta().abs().max() > 0.0

    def test_current_quaternions_remain_normalized(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(
            path,
            device=torch.device("cpu"),
            optimize_geometry=True,
            optimize_quats=True,
        )
        with torch.no_grad():
            assert scene.quat_delta_param is not None
            scene.quat_delta_param.fill_(0.1)

        norms = torch.linalg.vector_norm(scene.quats(), dim=-1)

        torch.testing.assert_close(norms, torch.ones_like(norms))

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for gsplat rasterization",
    )
    def test_render_batch_output_shape(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cuda:0"))
        c2w = torch.eye(4, device="cuda:0").unsqueeze(0)
        K = torch.tensor(
            [[500.0, 0.0, 200.0], [0.0, 500.0, 150.0], [0.0, 0.0, 1.0]],
            device="cuda:0",
        ).unsqueeze(0)
        rgb = scene.render_batch(c2w=c2w, K=K, width=400, height=300)
        assert rgb.shape == (1, 300, 400, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for gsplat rasterization",
    )
    def test_render_camera_view(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cuda:0"))
        camera = CameraView(
            c2w=torch.eye(4, device="cuda:0"),
            K=torch.tensor(
                [
                    [500.0, 0.0, 200.0],
                    [0.0, 500.0, 150.0],
                    [0.0, 0.0, 1.0],
                ],
                device="cuda:0",
            ),
            width=400,
            height=300,
            view_index=0,
        )
        rgb = scene.render(camera)
        assert rgb.shape == (3, 300, 400)

    def test_parameter_stats_keys(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cpu"))
        stats = scene.parameter_stats()
        expected_keys = {
            "appearance_rgb_min",
            "appearance_rgb_max",
            "appearance_rgb_under0_frac",
            "appearance_rgb_over1_frac",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_save_ply_materializes_scene_state(self, tmp_path: Path) -> None:
        path = _minimal_ply(tmp_path)
        scene = build_gaussian_scene(path, device=torch.device("cpu"))
        with torch.no_grad():
            scene.appearance_rgb.add_(0.1)
        out_path = tmp_path / "out.ply"

        scene.save_ply(out_path)

        saved = load_gaussian_ply(out_path)
        torch.testing.assert_close(saved["means"], scene.means(), rtol=0, atol=0)
        torch.testing.assert_close(saved["quats"], scene.quats(), rtol=0, atol=0)
        torch.testing.assert_close(saved["scales"], scene.scales_log(), rtol=0, atol=0)
        torch.testing.assert_close(
            saved["opacities"],
            scene.opacities_logit(),
            rtol=0,
            atol=0,
        )
        expected_sh0 = rgb_to_sh0(scene.colors()).unsqueeze(1).detach().cpu()
        torch.testing.assert_close(saved["sh0"], expected_sh0, rtol=0, atol=0)
