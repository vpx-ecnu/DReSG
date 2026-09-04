from pathlib import Path

import pytest

from dresg.evaluation.layout import (
    collect_method_layout,
    layout_key,
    parse_key_path,
    parse_methods,
)
from dresg.utils.results import (
    ViewManifestEntry,
    final_gaussians_path,
    final_video_path,
    load_view_manifest,
    write_view_manifest,
)


def test_layout_keys_are_exact_path_components() -> None:
    assert layout_key("018", name="Style") == "018"
    for invalid in ("", " 018", "018.png/other", ".", ".."):
        with pytest.raises(ValueError, match="path component"):
            layout_key(invalid, name="Style")
    with pytest.raises(TypeError, match="must be a string"):
        layout_key(18, name="Style")  # type: ignore[arg-type]


def test_parse_methods_rejects_empty_padded_or_duplicate_names() -> None:
    assert parse_methods("ours,baseline") == ("ours", "baseline")
    for invalid in ("", "ours,", "ours, baseline", "ours,ours"):
        with pytest.raises(ValueError):
            parse_methods(invalid)


def test_parse_key_path_requires_unique_exact_key_values() -> None:
    parsed = parse_key_path(["ours=/tmp/ours", "abc=/tmp/abc"])
    assert parsed["ours"] == Path("/tmp/ours")
    assert parsed["abc"] == Path("/tmp/abc")
    for invalid in (
        ["ours"],
        ["ours="],
        [" ours=/tmp/ours"],
        ["ours=/tmp/ours", "ours=/tmp/other"],
    ):
        with pytest.raises(ValueError):
            parse_key_path(invalid)


def test_collect_method_layout_ready(tmp_path: Path) -> None:
    root = tmp_path / "qualitative" / "ours" / "fern" / "018"
    renders = root / "renders"
    renders.mkdir(parents=True)
    (renders / "000000.png").touch()
    (renders / "000001.png").touch()
    write_view_manifest(
        root,
        [
            ViewManifestEntry(render_file="000000.png", view_index=0),
            ViewManifestEntry(render_file="000001.png", view_index=1),
        ],
    )
    content = [tmp_path / "content0.png", tmp_path / "content1.png"]

    layout = collect_method_layout(
        qual_root=tmp_path / "qualitative",
        method="ours",
        scene="fern",
        style="018",
        all_content_paths=content,
        view_mode="all",
    )

    assert layout.render_paths == (renders / "000000.png", renders / "000001.png")
    assert layout.content_paths == tuple(content)


def test_collect_method_layout_requires_manifest(tmp_path: Path) -> None:
    renders = tmp_path / "qualitative" / "ours" / "fern" / "018" / "renders"
    renders.mkdir(parents=True)
    (renders / "000000.png").touch()

    with pytest.raises(FileNotFoundError, match="Required view manifest"):
        collect_method_layout(
            qual_root=tmp_path / "qualitative",
            method="ours",
            scene="fern",
            style="018",
            all_content_paths=[tmp_path / "content0.png"],
            view_mode="all",
        )


def test_collect_method_layout_reports_view_mismatch(tmp_path: Path) -> None:
    renders = tmp_path / "qualitative" / "ours" / "fern" / "018" / "renders"
    renders.mkdir(parents=True)
    (renders / "000000.png").touch()
    write_view_manifest(
        renders.parent,
        [ViewManifestEntry(render_file="000000.png", view_index=0)],
    )
    content = [tmp_path / "content0.png", tmp_path / "content1.png"]

    with pytest.raises(ValueError, match="Expected view IDs 0..1"):
        collect_method_layout(
            qual_root=tmp_path / "qualitative",
            method="ours",
            scene="fern",
            style="018",
            all_content_paths=content,
            view_mode="all",
        )


def test_collect_method_layout_rendered_mode_truncates_content(tmp_path: Path) -> None:
    renders = tmp_path / "qualitative" / "ours" / "fern" / "018" / "renders"
    renders.mkdir(parents=True)
    (renders / "000000.png").touch()
    write_view_manifest(
        renders.parent,
        [ViewManifestEntry(render_file="000000.png", view_index=0)],
    )
    content = [tmp_path / "content0.png", tmp_path / "content1.png"]

    layout = collect_method_layout(
        qual_root=tmp_path / "qualitative",
        method="ours",
        scene="fern",
        style="018",
        all_content_paths=content,
        view_mode="rendered",
    )

    assert layout.content_paths == tuple(content[:1])


def test_collect_method_layout_uses_manifest_view_ids_not_filename_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qualitative" / "ours" / "fern" / "018"
    renders = root / "renders"
    renders.mkdir(parents=True)
    (renders / "left.png").touch()
    (renders / "right.png").touch()
    write_view_manifest(
        root,
        [
            ViewManifestEntry(render_file="left.png", view_index=1),
            ViewManifestEntry(render_file="right.png", view_index=0),
        ],
    )
    content = [tmp_path / "content0.png", tmp_path / "content1.png"]

    layout = collect_method_layout(
        qual_root=tmp_path / "qualitative",
        method="ours",
        scene="fern",
        style="018",
        all_content_paths=content,
        view_mode="all",
    )

    assert layout.render_paths == (renders / "right.png", renders / "left.png")
    assert layout.content_paths == tuple(content)
    assert layout.view_indices == (0, 1)


def test_final_artifact_paths_are_canonical(tmp_path: Path) -> None:
    root = tmp_path / "run"
    assert final_gaussians_path(root) == root / "point_cloud.ply"
    assert final_video_path(root) == root / "video.mp4"


def test_view_manifest_round_trip_is_immutable(tmp_path: Path) -> None:
    entries = (
        ViewManifestEntry(render_file="left.png", view_index=1),
        ViewManifestEntry(render_file="right.png", view_index=0),
    )

    write_view_manifest(tmp_path, entries)

    assert load_view_manifest(tmp_path) == entries


def test_write_view_manifest_rejects_noncanonical_entries(tmp_path: Path) -> None:
    for entry in (
        ViewManifestEntry(render_file=" view.png", view_index=0),
        ViewManifestEntry(render_file="nested/view.png", view_index=0),
        ViewManifestEntry(render_file="view.png", view_index=True),
    ):
        with pytest.raises((TypeError, ValueError)):
            write_view_manifest(tmp_path, [entry])


@pytest.mark.parametrize(
    "row",
    (
        " view.png,0",
        "nested/view.png,0",
        "view.png,00",
        "view.png,+1",
        "view.png, 1",
        "view.png,1,extra",
    ),
)
def test_load_view_manifest_rejects_noncanonical_values(
    tmp_path: Path,
    row: str,
) -> None:
    (tmp_path / "view_manifest.csv").write_text(
        f"render_file,view_index\n{row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_view_manifest(tmp_path)


def test_load_view_manifest_rejects_noncanonical_columns(tmp_path: Path) -> None:
    (tmp_path / "view_manifest.csv").write_text(
        "render_file,view_index,content_path\n000000.png,0,/private/content.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid view manifest columns"):
        load_view_manifest(tmp_path)


def test_load_view_manifest_rejects_duplicate_view_indices(tmp_path: Path) -> None:
    (tmp_path / "view_manifest.csv").write_text(
        "render_file,view_index\n000000.png,0\n000001.png,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate view_index"):
        load_view_manifest(tmp_path)
