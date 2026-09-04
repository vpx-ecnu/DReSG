from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from dresg.data.images import ViewImages
from dresg.training.optimization.gs import FitMetrics
from dresg.training.stages import ColorStage
from dresg.utils.color_transfer import rgb_covariance_match_colors, rgb_covariance_match_view_colors
from tests.config_factory import make_config


def _rgb_covariance_reference_match_colors(
    scene_images: torch.Tensor, style_image: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sh = scene_images.shape
    image_set = scene_images.view(-1, 3)
    style_img = style_image.view(-1, 3).to(image_set.device)
    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)
    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c).float() / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s).float() / float(style_img.size(0))
    u_c, sig_c, _ = torch.linalg.svd(cov_c)
    u_s, sig_s, _ = torch.linalg.svd(cov_s)
    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))
    tmp_mat = u_s @ scl_s @ u_s.transpose(1, 0) @ u_c @ scl_c @ u_c.transpose(1, 0)
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T
    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)
    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def _cov(pixels: torch.Tensor) -> torch.Tensor:
    centered = pixels - pixels.mean(0, keepdim=True)
    return centered.T @ centered / float(pixels.shape[0])


def test_rgb_covariance_match_colors_identity_is_nearly_unchanged() -> None:
    torch.manual_seed(1)
    source = torch.rand(3, 16, 16) * 0.5 + 0.25
    out, color_tf = rgb_covariance_match_colors(source, source)

    assert torch.allclose(out, source, atol=2e-4)
    assert color_tf.shape == (4, 4)
    assert torch.isfinite(color_tf).all()


def test_rgb_covariance_match_colors_matches_mean_and_covariance() -> None:
    torch.manual_seed(2)
    source = torch.rand(4, 3, 24, 24) * 0.35 + 0.25
    style = torch.rand(3, 32, 32) * 0.25 + 0.35
    out, _ = rgb_covariance_match_colors(source, style)
    out_pixels = out.permute(0, 2, 3, 1).reshape(-1, 3)
    style_pixels = style.permute(1, 2, 0).reshape(-1, 3)

    assert torch.allclose(out_pixels.mean(0), style_pixels.mean(0), atol=2e-3)
    assert torch.allclose(_cov(out_pixels), _cov(style_pixels), atol=3e-3)


def test_rgb_covariance_match_colors_is_finite_and_clamped_for_extreme_inputs() -> None:
    source = torch.zeros(3, 8, 8)
    style = torch.ones(3, 8, 8)
    out, color_tf = rgb_covariance_match_colors(source, style)

    assert torch.isfinite(out).all()
    assert torch.isfinite(color_tf).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_rgb_covariance_match_colors_matches_reference_formula() -> None:
    torch.manual_seed(3)
    source = torch.rand(2, 6, 5, 3)
    style = torch.rand(7, 4, 3)
    ref, ref_tf = _rgb_covariance_reference_match_colors(source, style)
    source_nchw = source.permute(0, 3, 1, 2)
    style_chw = style.permute(2, 0, 1)
    out, color_tf = rgb_covariance_match_colors(source_nchw, style_chw)

    assert torch.allclose(out.permute(0, 2, 3, 1), ref, atol=1e-6)
    assert torch.allclose(color_tf, ref_tf, atol=1e-6)


def test_rgb_covariance_match_view_colors_preserves_view_keys() -> None:
    images = {8: torch.rand(3, 4, 4), 2: torch.rand(3, 4, 4)}
    style = torch.rand(3, 5, 5)
    transferred, color_tf = rgb_covariance_match_view_colors(images, style)

    assert sorted(transferred) == [2, 8]
    assert transferred[2].shape == images[2].shape
    assert transferred[8].shape == images[8].shape
    assert color_tf.shape == (4, 4)


def test_post_color_transfer_calls_fit_and_returns_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    style_path = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), (180, 120, 80)).save(style_path)
    calls = {
        "fit": 0,
        "fit_steps": None,
        "update_geometry": None,
    }

    class FakeOptimizer:
        def run(self, **kwargs):
            calls["fit"] += 1
            calls["fit_steps"] = kwargs["fit_steps"]
            calls["update_geometry"] = kwargs["update_geometry"]
            return FitMetrics(
                final_total=0.25,
                final_l1=0.123,
                final_content3d_loss=0.0,
                elapsed_sec=0.5,
                fit_peak_allocated_mb=0.0,
            )

    class FakeScene:
        def render_current_images(self, cameras):
            return ViewImages(images_by_view={8: torch.full((3, 4, 4), 0.25)})

    tolist_calls = 0
    original_tolist = torch.Tensor.tolist

    def track_tolist(tensor):
        nonlocal tolist_calls
        tolist_calls += 1
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", track_tolist)
    config = make_config()
    metrics = ColorStage(
        FakeScene(),
        SimpleNamespace(c2w=torch.empty(0, device="cpu")),
        FakeOptimizer(),
        style_image=style_path,
        image_loss=config.image_loss,
    ).run(fit_steps=12)

    assert calls["fit"] == 1
    assert calls["fit_steps"] == 12
    assert calls["update_geometry"] is False
    assert metrics["post_color_transfer_enabled"] == 1
    assert metrics["post_color_transfer_fit_l1"] == 0.123
    assert tolist_calls == 1
