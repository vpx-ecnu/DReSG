from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dresg.inference.paths import (
    VideoPath,
    load_video_path,
    load_video_path_for_scene,
    save_video_path,
)
from dresg.inference.paths.trajectory import scene_fingerprint


def _path() -> VideoPath:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    c2w[:, 0, 3] = [-1.0, 0.0, 1.0]
    return VideoPath(
        c2w=c2w,
        K=np.array(
            [[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        width=100,
        height=80,
        scene_fingerprint="a" * 64,
        generation={
            "trajectory": "interpolated",
            "camera_source": "all",
            "n_frames": 3,
        },
    )


def test_video_path_codec_import_does_not_load_scipy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dresg.inference.paths.codec; "
            "assert 'scipy' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_video_path_codec_roundtrips_without_pickle(tmp_path: Path) -> None:
    output = tmp_path / "path.npz"

    save_video_path(output, _path())
    loaded = load_video_path(output)

    np.testing.assert_array_equal(loaded.c2w, _path().c2w)
    np.testing.assert_array_equal(loaded.K, _path().K)
    assert loaded.frame_count == 3
    assert loaded.width == 100
    assert loaded.height == 80
    assert dict(loaded.generation) == dict(_path().generation)
    assert not loaded.c2w.flags.writeable
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_video_path_codec_requires_exact_fields(tmp_path: Path) -> None:
    output = tmp_path / "invalid.npz"
    np.savez(
        output,
        format_version=np.asarray(1, dtype=np.int64),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match="Invalid video path fields"):
        load_video_path(output)


def test_video_path_codec_rejects_previous_format_version(
    tmp_path: Path,
) -> None:
    output = tmp_path / "path.npz"
    save_video_path(output, _path())
    with np.load(output, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["format_version"] = np.asarray(3, dtype=np.int64)
    np.savez(output, **payload)

    with pytest.raises(ValueError, match="format_version"):
        load_video_path(output)


def test_video_path_codec_rejects_duplicate_generation_keys(
    tmp_path: Path,
) -> None:
    output = tmp_path / "path.npz"
    save_video_path(output, _path())
    with np.load(output, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["generation_json"] = np.asarray(
        '{"trajectory":"interpolated","camera_source":"all",'
        '"n_frames":3,"n_frames":4}'
    )
    np.savez(output, **payload)

    with pytest.raises(ValueError, match="repeats key"):
        load_video_path(output)


def test_video_path_contract_rejects_noncanonical_pose_dtype() -> None:
    valid = _path()

    with pytest.raises(TypeError, match="float32"):
        VideoPath(
            c2w=valid.c2w.astype(np.float64),
            K=valid.K,
            width=valid.width,
            height=valid.height,
            scene_fingerprint=valid.scene_fingerprint,
            generation=valid.generation,
        )


@pytest.mark.parametrize(
    ("rotation", "message"),
    [
        (np.diag([2.0, 1.0, 1.0]), "uniform scale"),
        (np.diag([-1.0, 1.0, 1.0]), "right-handed"),
    ],
)
def test_video_path_rejects_invalid_rotations(
    rotation: np.ndarray,
    message: str,
) -> None:
    valid = _path()
    c2w = valid.c2w.copy()
    c2w[0, :3, :3] = rotation

    with pytest.raises(ValueError, match=message):
        replace(valid, c2w=c2w)


def test_video_path_accepts_a_right_handed_uniform_similarity_scale() -> None:
    valid = _path()
    c2w = valid.c2w.copy()
    c2w[:, :3, :3] *= 2.0

    scaled = replace(valid, c2w=c2w)

    np.testing.assert_array_equal(scaled.c2w, c2w)


def test_video_path_rejects_frame_varying_similarity_scale() -> None:
    valid = _path()
    c2w = valid.c2w.copy()
    c2w[0, :3, :3] *= 2.0

    with pytest.raises(ValueError, match="share one uniform scale"):
        replace(valid, c2w=c2w)


def test_video_path_rejects_noncanonical_intrinsics() -> None:
    valid = _path()
    K = valid.K.copy()
    K[2] = [1.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="canonical pinhole"):
        replace(valid, K=K)


def test_video_path_generation_must_match_pose_count() -> None:
    valid = _path()
    generation = dict(valid.generation)
    generation["n_frames"] = valid.frame_count + 1

    with pytest.raises(ValueError, match="n_frames"):
        replace(valid, generation=generation)


def test_video_path_generation_requires_trajectory_specific_fields() -> None:
    valid = _path()
    generation = dict(valid.generation)
    generation["ellipse_scale"] = 1.0

    with pytest.raises(ValueError, match="generation fields"):
        replace(valid, generation=generation)


class _Source(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.image_names)


def _source(scene_dir: Path, offset: float) -> _Source:
    c2w = np.eye(4, dtype=np.float64)[None]
    c2w[0, 0, 3] = offset
    return _Source(
        scene_dir=scene_dir,
        factor=4,
        image_names=("a.png",),
        camtoworlds=c2w,
        camera_ids=(1,),
        intrinsics_by_camera={1: np.eye(3, dtype=np.float64)},
        image_sizes_by_camera={1: (100, 80)},
    )


def test_scene_fingerprint_tracks_reconstruction_coordinates(
    tmp_path: Path,
) -> None:
    first = scene_fingerprint(
        _source(tmp_path, 0.0),
        trajectory="interpolated",
    )
    repeated = scene_fingerprint(
        _source(tmp_path, 0.0),
        trajectory="interpolated",
    )
    changed = scene_fingerprint(
        _source(tmp_path, 1.0),
        trajectory="interpolated",
    )

    assert first == repeated
    assert first != changed
    assert len(first) == 64


def test_llff_scene_fingerprint_tracks_forward_facing_bounds(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, 0.0)
    payload = np.zeros((1, 17), dtype=np.float64)
    payload[:, -2:] = [[0.1, 2.0]]
    np.save(tmp_path / "poses_bounds.npy", payload)
    first = scene_fingerprint(source, trajectory="llff_spiral")

    payload[:, -2:] = [[0.1, 3.0]]
    np.save(tmp_path / "poses_bounds.npy", payload)
    changed = scene_fingerprint(source, trajectory="llff_spiral")

    assert first != changed


def test_video_path_rejects_a_different_scene(tmp_path: Path) -> None:
    path = _path()
    matching_fingerprint = scene_fingerprint(
        _source(tmp_path, 0.0),
        trajectory="interpolated",
    )
    path = VideoPath(
        c2w=path.c2w,
        K=path.K,
        width=path.width,
        height=path.height,
        scene_fingerprint=matching_fingerprint,
        generation=path.generation,
    )
    output = tmp_path / "path.npz"
    save_video_path(output, path)

    with pytest.raises(ValueError, match="fingerprint"):
        load_video_path_for_scene(output, _source(tmp_path, 1.0))
