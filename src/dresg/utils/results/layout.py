"""Canonical result names and strict manifests shared by producers and consumers."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

RENDERS_DIRNAME = "renders"
VIEW_MANIFEST_FILENAME = "view_manifest.csv"
EXPORT_METRICS_FILENAME = "export_metrics.json"
FINAL_VIDEO_FILENAME = "video.mp4"
FINAL_GAUSSIANS_FILENAME = "point_cloud.ply"
VIEW_MANIFEST_FIELDS = (
    "render_file",
    "view_index",
)


@dataclass(frozen=True, slots=True)
class ViewManifestEntry:
    render_file: str
    view_index: int

    def to_row(self) -> dict[str, str | int]:
        return asdict(self)


def renders_dir(root: Path) -> Path:
    return root / RENDERS_DIRNAME


def view_manifest_path(root: Path) -> Path:
    return root / VIEW_MANIFEST_FILENAME


def export_metrics_path(root: Path) -> Path:
    return root / EXPORT_METRICS_FILENAME


def final_video_path(root: Path) -> Path:
    return root / FINAL_VIDEO_FILENAME


def final_gaussians_path(root: Path) -> Path:
    return root / FINAL_GAUSSIANS_FILENAME


def render_filename(view_index: int) -> str:
    return f"{view_index:06d}.png"


def _validate_manifest_entries(
    path: Path,
    entries: Sequence[ViewManifestEntry],
) -> None:
    if not entries:
        raise ValueError(f"{path} contains no view entries")
    for entry in entries:
        if not isinstance(entry, ViewManifestEntry):
            raise TypeError(f"{path} entries must be ViewManifestEntry values")
        render_file = entry.render_file
        if not isinstance(render_file, str):
            raise TypeError(f"{path} render_file values must be strings")
        if (
            not render_file
            or render_file != render_file.strip()
            or "/" in render_file
            or "\\" in render_file
            or render_file in {".", ".."}
        ):
            raise ValueError(f"{path} contains a noncanonical render_file: {render_file!r}")
        view_index = entry.view_index
        if isinstance(view_index, bool) or not isinstance(view_index, int):
            raise TypeError(f"{path} view_index values must be integers")
        if view_index < 0:
            raise ValueError(f"{path} contains a negative view_index")
    render_files = [entry.render_file for entry in entries]
    view_indices = [entry.view_index for entry in entries]
    if len(set(render_files)) != len(render_files):
        raise ValueError(f"{path} contains duplicate render_file values")
    if len(set(view_indices)) != len(view_indices):
        raise ValueError(f"{path} contains duplicate view_index values")


def write_view_manifest(
    root: Path,
    entries: Sequence[ViewManifestEntry],
) -> Path:
    path = view_manifest_path(root)
    _validate_manifest_entries(path, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIEW_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(entry.to_row() for entry in entries)
    return path


def _parse_view_index(value: str | None, *, path: Path, row_number: int) -> int:
    if value is None or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid view_index in {path} row {row_number}: {value!r}")
    view_index = int(value)
    if str(view_index) != value:
        raise ValueError(f"Noncanonical view_index in {path} row {row_number}: {value!r}")
    return view_index


def load_view_manifest(root: Path) -> tuple[ViewManifestEntry, ...]:
    path = view_manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Required view manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != VIEW_MANIFEST_FIELDS:
            raise ValueError(
                f"Invalid view manifest columns in {path}: "
                f"expected={list(VIEW_MANIFEST_FIELDS)}, actual={reader.fieldnames}"
            )
        entries = []
        for row_number, row in enumerate(reader, start=2):
            if set(row) != set(VIEW_MANIFEST_FIELDS):
                raise ValueError(f"Invalid fields in {path} row {row_number}")
            render_file = row["render_file"]
            if render_file is None:
                raise ValueError(
                    f"Invalid render_file in {path} row {row_number}: {render_file!r}"
                )
            entries.append(
                ViewManifestEntry(
                    render_file=render_file,
                    view_index=_parse_view_index(
                        row["view_index"],
                        path=path,
                        row_number=row_number,
                    ),
                )
            )
    _validate_manifest_entries(path, entries)
    return tuple(entries)
