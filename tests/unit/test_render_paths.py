from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dresg.inference.paths.geometry import (
    average_pose,
    focus_point,
    homogeneous_poses,
    normalize,
)
from dresg.inference.paths.llff import load_forward_facing_bounds
from dresg.inference.paths.trajectory import (
    VideoPathRequest,
    build_scene_video_poses,
    camera_scene_scale,
    video_camera_parameters,
)


def _request(**overrides) -> VideoPathRequest:
    trajectory = overrides.pop("trajectory", "interpolated")
    values = {
        "trajectory": trajectory,
        "camera_source": "all",
        "test_every": None,
        "n_frames": 7,
        "coord_mode": None,
        "llff_radius_scale": None,
        "ellipse_scale": None,
        "centered_llff_radius": None,
    }
    if trajectory == "llff_spiral":
        values.update(
            coord_mode="none",
            llff_radius_scale=1.0,
            centered_llff_radius=False,
        )
    elif trajectory == "builtin_spiral":
        values["llff_radius_scale"] = 1.0
    elif trajectory == "tnt_ellipse":
        values["ellipse_scale"] = 1.1
    values.update(overrides)
    return VideoPathRequest(**values)


@pytest.mark.parametrize(
    ("trajectory", "specific_fields"),
    [
        ("interpolated", set()),
        ("ellipse_z", set()),
        ("builtin_spiral", {"llff_radius_scale"}),
        ("tnt_ellipse", {"ellipse_scale"}),
        (
            "llff_spiral",
            {"coord_mode", "llff_radius_scale", "centered_llff_radius"},
        ),
    ],
)
def test_video_path_request_records_only_trajectory_inputs(
    trajectory: str,
    specific_fields: set[str],
) -> None:
    metadata = _request(trajectory=trajectory).metadata()

    assert set(metadata) == {
        "trajectory",
        "camera_source",
        "n_frames",
        *specific_fields,
    }


def test_video_path_request_rejects_irrelevant_parameters() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        _request(trajectory="interpolated", ellipse_scale=1.0)


def test_video_path_request_records_split_interval_only_for_train_split() -> None:
    request = _request(camera_source="train_split", test_every=8)

    assert request.metadata()["test_every"] == 8


@pytest.mark.parametrize(
    ("camera_source", "test_every", "message"),
    [
        ("train_split", None, "required"),
        ("all", 8, "does not apply"),
        ("interior", 8, "does not apply"),
    ],
)
def test_video_path_request_enforces_camera_source_parameters(
    camera_source: str,
    test_every: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(camera_source=camera_source, test_every=test_every)


def test_homogeneous_poses_supports_single_and_batched_poses() -> None:
    pose = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    single = homogeneous_poses(pose)
    batched = homogeneous_poses(np.stack([pose, pose + 1.0]))

    assert single.shape == (4, 4)
    assert batched.shape == (2, 4, 4)
    np.testing.assert_array_equal(single[-1], expected_bottom)
    np.testing.assert_array_equal(batched[:, -1], np.broadcast_to(expected_bottom, (2, 4)))


def test_homogeneous_poses_rejects_invalid_shape() -> None:
    invalid = np.zeros((2, 3, 3), dtype=np.float32)

    try:
        homogeneous_poses(invalid)
    except ValueError as error:
        assert "(3, 4) or (4, 4)" in str(error)
    else:
        raise AssertionError("Expected invalid pose shape to raise ValueError")


def test_focus_point_uses_stable_fallback_for_degenerate_axes() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = np.array([-1.0, 0.0, 1.0])

    point = focus_point(poses)

    assert point.shape == (3,)
    assert np.isfinite(point).all()
    np.testing.assert_allclose(point, np.zeros(3), atol=1e-8)


def test_normalize_rejects_degenerate_trajectory_vector() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        normalize(np.zeros(3, dtype=np.float64))


def test_average_pose_matches_mean_camera_frame() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[0, :3, 3] = np.array([-1.0, 0.0, 0.0])
    poses[1, :3, 3] = np.array([1.0, 0.0, 0.0])

    pose = average_pose(poses)

    assert pose.shape == (3, 4)
    np.testing.assert_allclose(pose[:, :3], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(pose[:, 3], np.zeros(3), atol=1e-8)


class _Source:
    def __init__(self, scene_dir: Path, camtoworlds: np.ndarray) -> None:
        self.scene_dir = scene_dir
        self.camtoworlds = camtoworlds
        self.camera_ids = tuple(0 for _ in camtoworlds)
        self.intrinsics_by_camera = {0: np.eye(3, dtype=np.float64)}
        self.image_sizes_by_camera = {0: (8, 6)}

    def __len__(self) -> int:
        return int(self.camtoworlds.shape[0])


def test_forward_facing_bounds_are_strict_and_scene_specific(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    source = _Source(tmp_path, poses)
    payload = np.zeros((2, 17), dtype=np.float64)
    payload[:, -2:] = [[0.1, 2.0], [0.2, 3.0]]
    np.save(tmp_path / "poses_bounds.npy", payload)

    bounds = load_forward_facing_bounds(source)

    np.testing.assert_allclose(bounds, [[0.1, 2.0], [0.2, 3.0]])
    assert not bounds.flags.writeable


def test_forward_facing_bounds_do_not_use_missing_file_fallback(tmp_path: Path) -> None:
    source = _Source(tmp_path, np.eye(4, dtype=np.float64)[None])

    with pytest.raises(FileNotFoundError, match="requires bounds file"):
        load_forward_facing_bounds(source)


def test_interpolated_path_has_exact_configured_frame_count(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = [-1.0, 0.0, 1.0]
    source = _Source(tmp_path, poses)
    path = build_scene_video_poses(source, _request(n_frames=7))

    assert path.shape == (7, 4, 4)
    assert np.isfinite(path).all()


def test_single_camera_interpolated_path_repeats_to_frame_count(
    tmp_path: Path,
) -> None:
    source = _Source(tmp_path, np.eye(4, dtype=np.float64)[None])
    path = build_scene_video_poses(source, _request(n_frames=5))

    assert path.shape == (5, 4, 4)
    np.testing.assert_array_equal(path, np.repeat(path[:1], 5, axis=0))


def test_video_camera_parameters_validate_only_selected_cameras(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 12, axis=0)
    source = _Source(tmp_path, poses)
    source.camera_ids = (1, *([0] * 11))
    source.intrinsics_by_camera = {
        0: np.eye(3, dtype=np.float64),
        1: np.diag([2.0, 2.0, 1.0]),
    }
    source.image_sizes_by_camera = {0: (8, 6), 1: (16, 12)}
    K, width, height = video_camera_parameters(
        source,
        camera_source="interior",
        test_every=None,
    )

    np.testing.assert_array_equal(K, np.eye(3))
    assert (width, height) == (8, 6)


def test_video_source_validation_fails_before_model_loading_without_bounds(
    tmp_path: Path,
) -> None:
    source = _Source(tmp_path, np.eye(4, dtype=np.float64)[None])
    with pytest.raises(FileNotFoundError, match="requires bounds file"):
        build_scene_video_poses(
            source,
            _request(trajectory="llff_spiral", n_frames=5),
        )


def test_forward_facing_bounds_reject_invalid_ranges(tmp_path: Path) -> None:
    source = _Source(tmp_path, np.eye(4, dtype=np.float64)[None])
    np.save(tmp_path / "poses_bounds.npy", np.array([[0.0, 0.0, 2.0, 1.0]]))

    with pytest.raises(ValueError, match="0 < near < far"):
        load_forward_facing_bounds(source)


def test_camera_scene_scale_uses_camera_centers() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[:, 0, 3] = [-2.0, 2.0]

    assert camera_scene_scale(_Source(Path("."), poses)) == pytest.approx(2.0)
