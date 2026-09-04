from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from dresg.models.gs.serialization import ply as ply_module
from dresg.models.gs.serialization.ply import load_gaussian_ply, save_gaussian_ply

_REQUIRED_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _write_gaussian_ply(
    path: Path,
    *,
    count: int = 2,
    normals: tuple[str, ...] = ("nx", "ny", "nz"),
    rest_names: tuple[str, ...] = (),
    omit: frozenset[str] = frozenset(),
    extras: tuple[str, ...] = (),
    field_dtypes: dict[str, str] | None = None,
    nonfinite_field: str | None = None,
    zero_quaternion: bool = False,
    text: bool = False,
    extra_element: bool = False,
    vertex_name: str = "vertex",
) -> None:
    dtypes = {} if field_dtypes is None else field_dtypes
    ordered_fields = (
        "x",
        "y",
        "z",
        *normals,
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *rest_names,
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        *extras,
    )
    field_names = tuple(name for name in ordered_fields if name not in omit)
    vertices = np.zeros(
        count,
        dtype=[(name, dtypes.get(name, "<f4")) for name in field_names],
    )
    if count:
        for offset, name in enumerate(("x", "y", "z")):
            if name in field_names:
                vertices[name] = np.arange(count, dtype=np.float32) + offset
        for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
            if name in field_names:
                vertices[name] = 0.5
        if "opacity" in field_names:
            vertices["opacity"] = np.arange(count, dtype=np.float32) / count
        if "rot_0" in field_names and not zero_quaternion:
            vertices["rot_0"] = 1.0
        for index, name in enumerate(rest_names):
            vertices[name] = float(index)
        if nonfinite_field is not None:
            vertices[nonfinite_field][0] = np.nan

    elements = [PlyElement.describe(vertices, vertex_name)]
    if extra_element:
        metadata = np.zeros(1, dtype=[("value", "<f4")])
        elements.append(PlyElement.describe(metadata, "metadata"))
    PlyData(elements, text=text, byte_order="<").write(str(path))


def _splats(count: int = 2) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])[:count],
        "quats": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
        )[:count],
        "scales": torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])[:count],
        "opacities": torch.tensor([0.25, -0.75])[:count],
        "sh0": torch.tensor([[[0.4, 0.5, 0.6]], [[-0.4, -0.5, -0.6]]])[:count],
    }


@pytest.mark.parametrize("rest_count", [0, 3, 9, 24, 45])
def test_load_gaussian_ply_discards_higher_order_sh(
    tmp_path: Path,
    rest_count: int,
) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(
        path,
        rest_names=tuple(f"f_rest_{index}" for index in range(rest_count)),
    )

    splats = load_gaussian_ply(path)

    assert set(splats) == {"means", "quats", "scales", "opacities", "sh0"}
    assert all(tensor.dtype == torch.float32 for tensor in splats.values())
    assert splats["means"].shape == (2, 3)
    assert splats["sh0"].shape == (2, 1, 3)
    torch.testing.assert_close(
        splats["sh0"],
        torch.full((2, 1, 3), 0.5),
        rtol=0,
        atol=0,
    )


def test_load_gaussian_ply_accepts_ascii_without_normals(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, normals=(), text=True)

    splats = load_gaussian_ply(path)

    assert splats["means"].shape == (2, 3)


def test_load_gaussian_ply_requires_all_scene_fields(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, omit=frozenset({"opacity"}))

    with pytest.raises(ValueError, match="missing required fields"):
        load_gaussian_ply(path)


def test_load_gaussian_ply_rejects_empty_scene(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, count=0)

    with pytest.raises(ValueError, match="at least one Gaussian"):
        load_gaussian_ply(path)


def test_load_gaussian_ply_discards_nonfinite_retained_rows(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, nonfinite_field="opacity")

    with pytest.warns(UserWarning, match="Discarded 1 Gaussian"):
        splats = load_gaussian_ply(path)

    assert splats["means"].shape == (1, 3)
    torch.testing.assert_close(
        splats["means"][0],
        torch.tensor([1.0, 2.0, 3.0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("field", ["nx", "f_rest_0"])
def test_load_gaussian_ply_ignores_nonfinite_unused_fields(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "scene.ply"
    rest_names = tuple(f"f_rest_{index}" for index in range(9))
    _write_gaussian_ply(path, rest_names=rest_names, nonfinite_field=field)

    splats = load_gaussian_ply(path)

    assert splats["means"].shape == (2, 3)


def test_load_gaussian_ply_rejects_scene_without_finite_rows(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, count=1, nonfinite_field="opacity")

    with pytest.raises(ValueError, match="no Gaussian with finite retained fields"):
        load_gaussian_ply(path)


def test_load_gaussian_ply_converts_consumed_fields_to_float32(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, field_dtypes={"x": "<f8", "opacity": "<f8"})

    splats = load_gaussian_ply(path)

    assert all(tensor.dtype == torch.float32 for tensor in splats.values())
    torch.testing.assert_close(
        splats["means"][:, 0],
        torch.tensor([0.0, 1.0]),
        rtol=0,
        atol=0,
    )


def test_load_gaussian_ply_rejects_zero_norm_quaternions(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, zero_quaternion=True)

    with pytest.raises(ValueError, match="non-zero norm"):
        load_gaussian_ply(path)


def test_load_gaussian_ply_ignores_unused_properties_and_elements(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(
        path,
        normals=("nx",),
        rest_names=("f_rest_0", "f_rest_2"),
        extras=("object_id",),
        extra_element=True,
    )

    splats = load_gaussian_ply(path)

    assert splats["means"].shape == (2, 3)


def test_load_gaussian_ply_requires_vertex_element(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_gaussian_ply(path, vertex_name="samples")

    with pytest.raises(ValueError, match="no vertex element"):
        load_gaussian_ply(path)


def test_ply_paths_require_lowercase_suffix(tmp_path: Path) -> None:
    path = tmp_path / "scene.bin"
    _write_gaussian_ply(path)

    with pytest.raises(ValueError, match="must use the .ply suffix"):
        load_gaussian_ply(path)
    with pytest.raises(ValueError, match="must use the .ply suffix"):
        save_gaussian_ply(path, splats=_splats())


def test_gaussian_ply_roundtrip_is_bitwise_exact(tmp_path: Path) -> None:
    splats = _splats()
    path = tmp_path / "scene.ply"

    save_gaussian_ply(path, splats=splats)
    recovered = load_gaussian_ply(path)
    ply = PlyData.read(str(path))
    fields = tuple(ply["vertex"].data.dtype.names or ())

    assert not ply.text
    assert ply.byte_order == "<"
    assert set(_REQUIRED_FIELDS).issubset(fields)
    assert all(name in fields for name in ("nx", "ny", "nz"))
    assert not any(name.startswith("f_rest_") for name in fields)
    assert all(np.all(ply["vertex"].data[name] == 0.0) for name in ("nx", "ny", "nz"))
    for name, tensor in splats.items():
        torch.testing.assert_close(recovered[name], tensor, rtol=0, atol=0)


def test_save_gaussian_ply_validates_canonical_splats(tmp_path: Path) -> None:
    invalid_keys = _splats()
    invalid_keys["shN"] = torch.zeros(2, 0, 3)
    with pytest.raises(ValueError, match="Invalid splat keys"):
        save_gaussian_ply(tmp_path / "extra-key.ply", splats=invalid_keys)

    invalid_dtype = _splats()
    invalid_dtype["means"] = invalid_dtype["means"].double()
    with pytest.raises(TypeError, match="splats.means must use torch.float32"):
        save_gaussian_ply(tmp_path / "float64.ply", splats=invalid_dtype)

    invalid_value = _splats()
    invalid_value["opacities"][0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_gaussian_ply(tmp_path / "nan.ply", splats=invalid_value)


def test_ply_write_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scene.ply"
    output.write_bytes(b"existing")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(ply_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_gaussian_ply(output, splats=_splats())

    assert output.read_bytes() == b"existing"
    assert not (tmp_path / ".scene.ply.tmp").exists()
