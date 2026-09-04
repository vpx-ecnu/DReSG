from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pycolmap
import pytest
from PIL import Image

from dresg.data.colmap import (
    ColmapScene,
    _load_reconstruction,
    _resolve_image_paths,
    load_colmap_scene,
)


def _touch_files(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _pose(tx: float) -> pycolmap.Rigid3d:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return pycolmap.Rigid3d(matrix)


def _reconstruction(
    *,
    images: dict[int, SimpleNamespace],
    cameras: dict[int, pycolmap.Camera],
    registered_ids: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        images=images,
        cameras=cameras,
        reg_image_ids=lambda: list(images) if registered_ids is None else registered_ids,
    )


def test_resolve_image_paths_matches_relative_names(tmp_path: Path) -> None:
    pyramid = tmp_path / "images_4"
    names = ["nested/a.png", "nested/b.png"]
    _touch_files(pyramid, names)

    paths = _resolve_image_paths(
        image_names=["nested/b.png", "nested/a.png"],
        original_dir=tmp_path / "images",
        pyramid_dir=pyramid,
    )

    assert paths == (pyramid / "nested/b.png", pyramid / "nested/a.png")


def test_resolve_image_paths_allows_extension_changes_with_matching_stems(
    tmp_path: Path,
) -> None:
    pyramid = tmp_path / "images_4"
    _touch_files(pyramid, ["IMG_0002.png", "IMG_0001.png"])

    paths = _resolve_image_paths(
        image_names=["IMG_0002.JPG", "IMG_0001.JPG"],
        original_dir=tmp_path / "images",
        pyramid_dir=pyramid,
    )

    assert paths == (pyramid / "IMG_0002.png", pyramid / "IMG_0001.png")


def test_resolve_image_paths_detects_llff_indexed_names(tmp_path: Path) -> None:
    original = tmp_path / "images"
    pyramid = tmp_path / "images_4"
    _touch_files(original, ["IMG_0002.JPG", "IMG_0001.JPG"])
    _touch_files(pyramid, ["image001.png", "image000.png"])

    paths = _resolve_image_paths(
        image_names=["IMG_0002.JPG", "IMG_0001.JPG"],
        original_dir=original,
        pyramid_dir=pyramid,
    )

    assert paths == (pyramid / "image001.png", pyramid / "image000.png")


def test_resolve_image_paths_rejects_unrecognized_sorted_pairing(
    tmp_path: Path,
) -> None:
    original = tmp_path / "images"
    pyramid = tmp_path / "images_4"
    _touch_files(original, ["IMG_0002.JPG", "IMG_0001.JPG"])
    _touch_files(pyramid, ["frame001.png", "frame000.png"])

    with pytest.raises(ValueError, match="must preserve COLMAP relative names or use"):
        _resolve_image_paths(
            image_names=["IMG_0002.JPG", "IMG_0001.JPG"],
            original_dir=original,
            pyramid_dir=pyramid,
        )


def test_resolve_image_paths_rejects_duplicate_colmap_names(tmp_path: Path) -> None:
    pyramid = tmp_path / "images"
    _touch_files(pyramid, ["a.png"])

    with pytest.raises(ValueError, match="must be unique"):
        _resolve_image_paths(
            image_names=["a.png", "a.png"],
            original_dir=pyramid,
            pyramid_dir=pyramid,
        )


def test_resolve_image_paths_rejects_duplicate_colmap_stems(tmp_path: Path) -> None:
    pyramid = tmp_path / "images"
    _touch_files(pyramid, ["a.png"])

    with pytest.raises(ValueError, match="unique relative stems"):
        _resolve_image_paths(
            image_names=["a.jpg", "a.png"],
            original_dir=pyramid,
            pyramid_dir=pyramid,
        )


def test_load_reconstruction_rejects_unrelated_pycolmap(monkeypatch) -> None:
    monkeypatch.delattr("dresg.data.colmap.pycolmap.Reconstruction", raising=False)

    with pytest.raises(ImportError, match="official pycolmap"):
        _load_reconstruction(Path("/tmp/colmap"))


def test_load_colmap_scene_uses_registered_images_and_canonical_filename_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (8, 6), "red").save(images_dir / "b.png")
    Image.new("RGB", (8, 6), "blue").save(images_dir / "a.png")
    Image.new("RGB", (8, 6), "green").save(images_dir / "unused.png")

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=8,
        height=6,
        params=[4.0, 5.0, 3.0, 2.0],
    )
    reconstruction = _reconstruction(
        images={
            7: SimpleNamespace(
                name="b.png",
                camera_id=1,
                cam_from_world=lambda: _pose(2.0),
            ),
            3: SimpleNamespace(
                name="a.png",
                camera_id=1,
                cam_from_world=lambda: _pose(1.0),
            ),
            9: SimpleNamespace(
                name="unused.png",
                camera_id=1,
                cam_from_world=lambda: _pose(3.0),
            ),
        },
        cameras={1: camera},
        registered_ids=[7, 3],
    )
    monkeypatch.setattr(
        "dresg.data.colmap.pycolmap.Reconstruction",
        lambda _path: reconstruction,
        raising=False,
    )

    source = load_colmap_scene(
        scene_dir=tmp_path,
        factor=1,
    )

    assert isinstance(source, ColmapScene)
    assert source.image_names == ("a.png", "b.png")
    assert source.image_paths == (images_dir / "a.png", images_dir / "b.png")
    assert source.camera_ids == (1, 1)
    assert source.image_sizes_by_camera[1] == (8, 6)
    with pytest.raises(TypeError, match="Camera view index"):
        source.validate_view_index(True)
    np.testing.assert_allclose(
        source.intrinsics_by_camera[1],
        [[4.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(source.camtoworlds[:, 0, 3], [-1.0, -2.0])
    with pytest.raises(ValueError, match="read-only"):
        source.camtoworlds[0, 0, 0] = 2.0
    with pytest.raises(TypeError):
        source.intrinsics_by_camera[1] = np.eye(3)  # type: ignore[index]


def test_load_colmap_scene_rescales_intrinsics_to_realized_odd_dimensions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    (tmp_path / "images").mkdir()
    pyramid = tmp_path / "images_2"
    pyramid.mkdir()
    Image.new("RGB", (101, 77), "red").save(tmp_path / "images" / "odd.png")
    Image.new("RGB", (50, 38), "red").save(pyramid / "odd.png")

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=101,
        height=77,
        params=[100.0, 80.0, 50.0, 40.0],
    )
    reconstruction = _reconstruction(
        images={
            1: SimpleNamespace(
                name="odd.png",
                camera_id=5,
                cam_from_world=lambda: _pose(0.0),
            )
        },
        cameras={5: camera},
    )
    monkeypatch.setattr(
        "dresg.data.colmap.pycolmap.Reconstruction",
        lambda _path: reconstruction,
        raising=False,
    )

    source = load_colmap_scene(
        scene_dir=tmp_path,
        factor=2,
    )

    K = source.intrinsics_by_camera[5]
    assert source.image_sizes_by_camera[5] == (50, 38)
    assert K[0, 0] == pytest.approx(100.0 * 50.0 / 101.0)
    assert K[1, 1] == pytest.approx(80.0 * 38.0 / 77.0)
    assert K[0, 2] == pytest.approx(50.0 * 50.0 / 101.0)
    assert K[1, 2] == pytest.approx(40.0 * 38.0 / 77.0)


def test_load_colmap_scene_rejects_invalid_factor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="factor must be positive"):
        load_colmap_scene(
            scene_dir=tmp_path,
            factor=0,
        )


def test_load_colmap_scene_rejects_input_coercion(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="scene_dir must be a pathlib.Path"):
        load_colmap_scene(
            scene_dir=str(tmp_path),
            factor=1,
        )
    with pytest.raises(TypeError, match="factor must be an integer"):
        load_colmap_scene(
            scene_dir=tmp_path,
            factor=True,
        )
