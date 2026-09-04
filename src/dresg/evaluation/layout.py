"""Qualitative result layout helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dresg.data.colmap import load_colmap_scene
from dresg.utils.results import load_view_manifest, renders_dir

DEFAULT_METHODS = ("ours",)


@dataclass(frozen=True, slots=True)
class MethodLayout:
    root: Path
    render_paths: tuple[Path, ...]
    content_paths: tuple[Path, ...]
    view_indices: tuple[int, ...]


def layout_key(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component")
    return value


def parse_methods(spec: str) -> tuple[str, ...]:
    methods = tuple(
        layout_key(method, name="Method")
        for method in spec.split(",")
    )
    if len(set(methods)) != len(methods):
        raise ValueError("Methods must be unique")
    return methods


def parse_key_path(items: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected method=path, got: {item}")
        key, value = item.split("=", 1)
        key = layout_key(key, name="Result method")
        if key in result:
            raise ValueError(f"Result method is repeated: {key}")
        if not value or value != value.strip():
            raise ValueError(f"Result path must not be empty or padded: {item}")
        result[key] = Path(value).expanduser()
    return result


def content_image_paths(
    scene_dir: Path,
    factor: int,
) -> tuple[Path, ...]:
    source = load_colmap_scene(
        scene_dir=scene_dir,
        factor=factor,
    )
    return source.image_paths


def _manifest_view_indices(
    root: Path,
    render_paths: Sequence[Path],
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    entries = load_view_manifest(root)
    by_name = {entry.render_file: entry.view_index for entry in entries}
    missing = [path.name for path in render_paths if path.name not in by_name]
    extras = sorted(set(by_name) - {path.name for path in render_paths})
    if missing or extras:
        raise ValueError(f"view manifest mismatch: missing={missing}, extras={extras}")
    indexed = sorted(
        ((by_name[path.name], path) for path in render_paths),
        key=lambda item: item[0],
    )
    return (
        tuple(path for _view_index, path in indexed),
        tuple(view_index for view_index, _path in indexed),
    )


def collect_method_layout(
    *,
    qual_root: Path,
    method: str,
    scene: str,
    style: str,
    all_content_paths: Sequence[Path],
    view_mode: str,
    run_root: Path | None = None,
) -> MethodLayout:
    method = layout_key(method, name="Method")
    scene = layout_key(scene, name="Scene")
    style = layout_key(style, name="Style")
    root = run_root if run_root is not None else qual_root / method / scene / style
    if view_mode not in {"all", "rendered"}:
        raise ValueError(f"Unsupported view_mode: {view_mode}")
    method_renders_dir = renders_dir(root)
    render_paths = tuple(sorted(method_renders_dir.glob("*.png")))
    if not render_paths:
        raise FileNotFoundError(f"No PNG renders found under {method_renders_dir}")
    render_paths, view_indices = _manifest_view_indices(root, render_paths)
    invalid = [index for index in view_indices if index >= len(all_content_paths)]
    if invalid:
        raise ValueError(f"View indices outside content range: {invalid}")
    if view_mode == "all" and view_indices != tuple(range(len(all_content_paths))):
        raise ValueError(
            f"Expected view IDs 0..{len(all_content_paths) - 1}, got {list(view_indices)}"
        )
    content_paths = tuple(all_content_paths[index] for index in view_indices)

    return MethodLayout(
        root=root,
        render_paths=render_paths,
        content_paths=content_paths,
        view_indices=view_indices,
    )
