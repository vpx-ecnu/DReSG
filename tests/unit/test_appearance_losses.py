from __future__ import annotations

import pytest
import torch

from dresg.config import ImageLossConfig
from dresg.data.cameras import Cameras, CameraView
from dresg.models.gs.fitting import AppearanceLosses, compute_appearance_losses
from dresg.training.optimization.gs import FitMetrics, GaussianOptimizer
from tests.config_factory import make_config


def camera() -> CameraView:
    return CameraView(
        view_index=0,
        c2w=torch.eye(4),
        K=torch.eye(3),
        width=8,
        height=8,
    )


def image_loss(
    *,
    content3d: float = 0.0,
    l1: float = 0.0,
    l1_use_unclamped_render: bool = True,
) -> ImageLossConfig:
    return ImageLossConfig(
        lambda_l1=l1,
        l1_use_unclamped_render=l1_use_unclamped_render,
        lambda_dssim=0.0,
        lambda_img_tv=0.0,
        lambda_content3d=content3d,
        content3d_dino_model="facebook/dinov2-base",
        content3d_dino_size=224,
    )


def loss_for(
    *,
    render: torch.Tensor,
    teacher: torch.Tensor | None = None,
    base_render: torch.Tensor | None = None,
    content3d: float = 0.0,
    content_loss=None,
) -> AppearanceLosses:
    return compute_appearance_losses(
        render_rgb=render,
        teacher_rgb=teacher if teacher is not None else render.detach().clone(),
        base_render_rgb=base_render,
        content_loss=content_loss,
        view_id=0,
        config=image_loss(content3d=content3d),
    )


def test_zero_content3d_weight_disables_content_loss() -> None:
    render = torch.zeros(3, 8, 8)
    teacher = torch.ones(3, 8, 8)
    base_render = torch.zeros(3, 8, 8)

    losses_no_target = loss_for(
        render=render,
        teacher=teacher,
        base_render=None,
    )
    losses_with_target = loss_for(
        render=render,
        teacher=teacher,
        base_render=base_render,
        content3d=0.0,
    )

    torch.testing.assert_close(losses_with_target.total, losses_no_target.total)
    assert float(losses_with_target.content3d.item()) == pytest.approx(0.0)


def test_appearance_losses_are_typed_and_differentiable() -> None:
    render = torch.zeros(3, 8, 8, requires_grad=True)

    losses = loss_for(render=render)

    assert isinstance(losses, AppearanceLosses)
    assert losses.total.ndim == 0
    losses.total.backward()
    assert render.grad is not None


def test_appearance_losses_require_matching_teacher_domain() -> None:
    render = torch.zeros(3, 8, 8)

    with pytest.raises(ValueError, match="shape must match"):
        loss_for(render=render, teacher=torch.zeros(3, 7, 8))
    with pytest.raises(ValueError, match="device and dtype"):
        loss_for(render=render, teacher=torch.zeros(3, 8, 8, dtype=torch.float64))


def test_l1_can_follow_unclamped_paper_render_while_aux_losses_use_clamped_rgb() -> None:
    render = torch.full((3, 2, 2), 2.0)
    teacher = torch.zeros_like(render)
    raw_losses = compute_appearance_losses(
        render_rgb=render,
        teacher_rgb=teacher,
        base_render_rgb=None,
        content_loss=None,
        view_id=0,
        config=image_loss(l1=1.0, l1_use_unclamped_render=True),
    )
    clamped_losses = compute_appearance_losses(
        render_rgb=render,
        teacher_rgb=teacher,
        base_render_rgb=None,
        content_loss=None,
        view_id=0,
        config=image_loss(l1=1.0, l1_use_unclamped_render=False),
    )

    assert float(raw_losses.l1.item()) == pytest.approx(2.0)
    assert float(clamped_losses.l1.item()) == pytest.approx(1.0)


def test_appearance_fit_metrics_include_core_fit_fields() -> None:
    payload = FitMetrics(
        final_total=0.5,
        final_l1=0.25,
        final_content3d_loss=0.123,
        elapsed_sec=2.0,
        fit_peak_allocated_mb=12.0,
    ).to_dict()

    assert payload["final_content3d_loss"] == pytest.approx(0.123)
    assert payload["fit_peak_allocated_mb"] == pytest.approx(12.0)


def test_dino_patch_content_loss_uses_run_owned_loss() -> None:
    render = torch.zeros(3, 8, 8)
    base_render = torch.ones(3, 8, 8)

    class FakeContentLoss:
        def loss(
            self,
            render_bchw: torch.Tensor,
            base_render_bchw: torch.Tensor,
            *,
            view_id: int,
        ) -> torch.Tensor:
            assert render_bchw.shape == base_render_bchw.shape
            assert view_id == 0
            return render_bchw.new_tensor(0.25)

    losses = loss_for(
        render=render,
        base_render=base_render,
        content3d=0.4,
        content_loss=FakeContentLoss(),
    )

    assert float(losses.content3d.item()) == pytest.approx(0.25)
    assert float(losses.total.item()) == pytest.approx(0.1)


def test_enabled_content_loss_rejects_partial_inputs() -> None:
    render = torch.zeros(3, 8, 8)
    base_render = torch.ones(3, 8, 8)

    class FakeContentLoss:
        def loss(
            self,
            render_bchw: torch.Tensor,
            base_render_bchw: torch.Tensor,
            *,
            view_id: int,
        ) -> torch.Tensor:
            raise AssertionError("Incomplete content inputs must fail before evaluation")

    with pytest.raises(ValueError, match="requires both"):
        loss_for(
            render=render,
            base_render=None,
            content3d=1.0,
            content_loss=FakeContentLoss(),
        )
    with pytest.raises(ValueError, match="requires both"):
        loss_for(
            render=render,
            base_render=base_render,
            content3d=1.0,
            content_loss=None,
        )



def test_appearance_fit_reads_device_metrics_only_at_final_step(monkeypatch) -> None:
    class AppearanceOnlyScene:
        def __init__(self) -> None:
            self.appearance = torch.nn.Parameter(torch.tensor(0.1))

        def appearance_parameters(self):
            return [self.appearance]

        def geometry_parameters(self):
            return []

        def geometry_mean_scale_parameters(self):
            return []

        def geometry_quat_parameters(self):
            return []

        def render(self, _camera: CameraView, *, clamp: bool = False):
            image = self.appearance.expand(3, 2, 2)
            return image.clamp(0.0, 1.0) if clamp else image

        def apply_parameter_constraints(self) -> None:
            with torch.no_grad():
                self.appearance.clamp_(0.0, 1.0)

    scene = AppearanceOnlyScene()
    cameras = Cameras(
        view_indices=(0,),
        c2w=torch.eye(4).unsqueeze(0),
        K=torch.eye(3).unsqueeze(0),
        width=2,
        height=2,
    )
    tolist_calls = 0
    original_tolist = torch.Tensor.tolist

    def track_tolist(tensor):
        nonlocal tolist_calls
        tolist_calls += 1
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", track_tolist)
    config = make_config()
    config.appearance_optim.lr = 0.1
    optimizer = GaussianOptimizer(scene, cameras, config.appearance_optim)
    optimizer.run(
        teachers_by_view={0: torch.ones(3, 2, 2)},
        base_renders_by_view=None,
        image_loss=image_loss(l1=1.0),
        content_loss=None,
        fit_steps=5,
        appearance_update_rule="standard",
        update_geometry=True,
    )

    assert tolist_calls == 1


def test_appearance_fit_can_freeze_geometry_for_post_color_transfer() -> None:
    class TrainableScene:
        def __init__(self) -> None:
            self.appearance = torch.nn.Parameter(torch.tensor(0.1))
            self.geometry = torch.nn.Parameter(torch.tensor(0.2))

        def appearance_parameters(self) -> list[torch.nn.Parameter]:
            return [self.appearance]

        def geometry_parameters(self) -> list[torch.nn.Parameter]:
            return [self.geometry]

        def geometry_mean_scale_parameters(self) -> list[torch.nn.Parameter]:
            return [self.geometry]

        def geometry_quat_parameters(self) -> list[torch.nn.Parameter]:
            return []

        def render(self, _camera: CameraView, *, clamp: bool = False) -> torch.Tensor:
            image = (self.appearance + self.geometry).expand(3, 2, 2)
            return image.clamp(0.0, 1.0) if clamp else image

        def apply_parameter_constraints(self) -> None:
            with torch.no_grad():
                self.appearance.clamp_(0.0, 1.0)

    scene = TrainableScene()
    camera = CameraView(
        view_index=0,
        c2w=torch.eye(4),
        K=torch.eye(3),
        width=2,
        height=2,
    )
    cameras = Cameras(
        view_indices=(camera.view_index,),
        c2w=camera.c2w.unsqueeze(0),
        K=camera.K.unsqueeze(0),
        width=camera.width,
        height=camera.height,
    )
    config = make_config()
    config.appearance_optim.lr = 0.1
    config.appearance_optim.geometry_lr = 0.1
    optimizer = GaussianOptimizer(scene, cameras, config.appearance_optim)
    geometry_before = scene.geometry.detach().clone()

    optimizer.run(
        teachers_by_view={0: torch.ones(3, 2, 2)},
        base_renders_by_view=None,
        image_loss=image_loss(l1=1.0),
        content_loss=None,
        fit_steps=2,
        appearance_update_rule="standard",
        update_geometry=False,
    )

    assert torch.equal(scene.geometry.detach(), geometry_before)
    assert scene.geometry.requires_grad
    assert scene.appearance.detach() > 0.1
