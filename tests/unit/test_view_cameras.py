from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dresg.data.cameras import Cameras, CameraView, build_scaled_cameras, scaled_intrinsics


def _source(
    *,
    widths: tuple[int, ...],
    heights: tuple[int, ...],
) -> SimpleNamespace:
    count = len(widths)
    source = SimpleNamespace(
        camera_ids=tuple(range(count)),
        intrinsics_by_camera={
            index: np.eye(3, dtype=np.float64) for index in range(count)
        },
        camtoworlds=np.stack(
            [np.eye(4, dtype=np.float64) * float(index + 1) for index in range(count)]
        ),
        image_sizes_by_camera={
            index: (widths[index], heights[index]) for index in range(count)
        },
    )

    def validate_view_index(view_index: int) -> None:
        if view_index < 0 or view_index >= count:
            raise ValueError(
                f"Camera view index {view_index} is outside the valid range 0..{count - 1}"
            )

    source.validate_view_index = validate_view_index
    return source


def test_scaled_intrinsics_uses_realized_noninteger_dimensions() -> None:
    K = torch.tensor(
        [
            [100.0, 0.0, 50.0],
            [0.0, 80.0, 40.0],
            [0.0, 0.0, 1.0],
        ]
    )

    scaled_K, width, height = scaled_intrinsics(
        K,
        width=101,
        height=77,
        scale=0.5,
    )

    assert (width, height) == (50, 38)
    assert scaled_K[0, 0] == pytest.approx(100.0 * 50.0 / 101.0)
    assert scaled_K[1, 1] == pytest.approx(80.0 * 38.0 / 77.0)
    assert scaled_K[0, 2] == pytest.approx(50.0 * 50.0 / 101.0)
    assert scaled_K[1, 2] == pytest.approx(40.0 * 38.0 / 77.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_scaled_intrinsics_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        scaled_intrinsics(torch.eye(3), width=10, height=10, scale=scale)


def test_build_scaled_cameras_checks_resolution_and_batches_tensors() -> None:
    source = _source(widths=(50, 200, 200), heights=(25, 100, 100))

    cameras = build_scaled_cameras(
        source=source,
        view_ids=[1, 2],
        device=torch.device("cpu"),
        render_scale=0.5,
        reference_width=100,
        reference_height=50,
        label="aux",
    )

    assert isinstance(cameras, Cameras)
    assert cameras.width == 100
    assert cameras.height == 50
    assert cameras.view_indices == (1, 2)
    assert cameras.c2w.shape == (2, 4, 4)
    assert cameras.K.shape == (2, 3, 3)
    assert [item.view_index for item in cameras] == [1, 2]
    first = cameras.view(0)
    assert isinstance(first, CameraView)
    assert torch.equal(first.c2w, torch.eye(4) * 2.0)
    assert first.width == 100
    assert first.height == 50


def test_build_scaled_cameras_raises_on_mismatched_resolution() -> None:
    source = _source(widths=(50, 200), heights=(25, 120))

    with pytest.raises(ValueError, match="All aux camera resolutions"):
        build_scaled_cameras(
            source=source,
            view_ids=[1],
            device=torch.device("cpu"),
            render_scale=0.5,
            reference_width=100,
            reference_height=50,
            label="aux",
        )


def test_build_scaled_cameras_requires_complete_reference_resolution() -> None:
    with pytest.raises(ValueError, match="provided together"):
        build_scaled_cameras(
            source=_source(widths=(100,), heights=(50,)),
            view_ids=[0],
            device=torch.device("cpu"),
            render_scale=1.0,
            reference_width=100,
            reference_height=None,
            label="aux",
        )


def test_build_scaled_cameras_handles_empty_view_list() -> None:
    cameras = build_scaled_cameras(
        source=SimpleNamespace(),
        view_ids=[],
        device=torch.device("cpu"),
        render_scale=0.5,
        reference_width=100,
        reference_height=50,
        label="aux",
    )

    assert len(cameras) == 0
    assert cameras.view_indices == ()
    assert cameras.c2w.shape == (0, 4, 4)
    assert cameras.K.shape == (0, 3, 3)


def test_cameras_reject_duplicate_view_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        Cameras(
            view_indices=(1, 1),
            c2w=torch.eye(4).repeat(2, 1, 1),
            K=torch.eye(3).repeat(2, 1, 1),
            width=10,
            height=10,
        )


def test_camera_domains_reject_boolean_identifiers_and_dimensions() -> None:
    with pytest.raises(TypeError, match="view_indices entries"):
        Cameras(
            view_indices=(False,),
            c2w=torch.eye(4).unsqueeze(0),
            K=torch.eye(3).unsqueeze(0),
            width=10,
            height=10,
        )
    with pytest.raises(TypeError, match="Camera width"):
        CameraView(
            view_index=0,
            c2w=torch.eye(4),
            K=torch.eye(3),
            width=True,
            height=10,
        )


def test_camera_batch_rejects_nonfinite_tensors() -> None:
    c2w = torch.eye(4).unsqueeze(0)
    c2w[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="only finite"):
        Cameras(
            view_indices=(0,),
            c2w=c2w,
            K=torch.eye(3).unsqueeze(0),
            width=10,
            height=10,
        )


def test_camera_lookup_rejects_boolean_and_out_of_range_positions() -> None:
    cameras = Cameras(
        view_indices=(3,),
        c2w=torch.eye(4).unsqueeze(0),
        K=torch.eye(3).unsqueeze(0),
        width=10,
        height=10,
    )

    with pytest.raises(TypeError, match="Camera position"):
        cameras.view(True)
    with pytest.raises(IndexError, match="outside"):
        cameras.view(1)


def test_scaled_camera_inputs_are_not_coerced() -> None:
    with pytest.raises(TypeError, match="Camera scale must be numeric"):
        scaled_intrinsics(torch.eye(3), width=10, height=10, scale=True)
    with pytest.raises(TypeError, match="Camera width"):
        scaled_intrinsics(torch.eye(3), width=False, height=10, scale=1.0)
    with pytest.raises(TypeError, match="view_ids entries"):
        build_scaled_cameras(
            source=_source(widths=(10,), heights=(10,)),
            view_ids=[False],
            device=torch.device("cpu"),
            render_scale=1.0,
            label="strict",
        )
