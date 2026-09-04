from __future__ import annotations

import torch

from dresg.models.gs.rendering.rasterization import rasterize_gaussians


def test_rasterize_gaussians_prepares_canonical_gsplat_inputs() -> None:
    calls = []

    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        renders = torch.tensor(
            [
                [
                    [[-0.5, 0.2, 1.5, 2.0], [0.3, 0.4, 0.5, 3.0]],
                    [[0.6, 0.7, 0.8, 4.0], [1.2, -0.1, 0.9, 5.0]],
                ]
            ],
            dtype=torch.float32,
        )
        alphas = torch.tensor(
            [[[[0.1], [0.2]], [[0.3], [0.4]]]],
            dtype=torch.float32,
        )
        info = {"gaussian_ids": torch.tensor([1, 2])}
        return renders, alphas, info

    means = torch.zeros((2, 3))
    quats = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    scales_log = torch.log(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    opacities_logit = torch.tensor([0.0, 1.0])
    colors = torch.cat(
        [torch.ones((2, 1, 3)), torch.zeros((2, 2, 3))],
        dim=1,
    )
    c2w = torch.eye(4).unsqueeze(0)
    K = torch.eye(3).unsqueeze(0)

    renders, alphas, info = rasterize_gaussians(
        means=means,
        quats=quats,
        scales_log=scales_log,
        opacities_logit=opacities_logit,
        colors=colors,
        c2w=c2w,
        K=K,
        width=2,
        height=2,
        packed=True,
        render_mode="RGB+ED",
        sh_degree=0,
        rasterization_fn=fake_rasterization,
    )

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["means"] is means
    assert torch.allclose(
        kwargs["quats"],
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.allclose(kwargs["scales"], torch.exp(scales_log))
    assert torch.allclose(kwargs["opacities"], torch.sigmoid(opacities_logit))
    assert kwargs["colors"] is colors
    assert torch.equal(kwargs["viewmats"], torch.eye(4).unsqueeze(0))
    assert kwargs["Ks"] is K
    assert kwargs["width"] == 2
    assert kwargs["height"] == 2
    assert kwargs["packed"] is True
    assert kwargs["absgrad"] is False
    assert kwargs["sparse_grad"] is False
    assert kwargs["rasterize_mode"] == "classic"
    assert kwargs["distributed"] is False
    assert kwargs["camera_model"] == "pinhole"
    assert kwargs["render_mode"] == "RGB+ED"
    assert kwargs["sh_degree"] == 0
    assert renders.shape == (1, 2, 2, 4)
    assert alphas.shape == (1, 2, 2, 1)
    assert torch.equal(info["gaussian_ids"], torch.tensor([1, 2]))
