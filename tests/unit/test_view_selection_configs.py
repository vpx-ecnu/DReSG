from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "conf" / "view_selection"

EXPECTED_COUNTS = {
    ("llff", "fern"): (20, 17),
    ("llff", "flower"): (34, 23),
    ("llff", "fortress"): (42, 18),
    ("llff", "horns"): (62, 29),
    ("llff", "leaves"): (26, 24),
    ("llff", "orchids"): (25, 25),
    ("llff", "trex"): (55, 21),
    ("tnt", "family"): (152, 77),
    ("tnt", "m60"): (313, 60),
    ("tnt", "playground"): (307, 55),
    ("tnt", "train"): (301, 66),
    ("tnt", "truck"): (251, 67),
}


def test_paper_view_selection_presets_are_complete_and_valid() -> None:
    discovered: set[tuple[str, str]] = set()

    for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
        payload = yaml.safe_load(path.read_text())
        metadata = payload["view_selection"]
        views = payload["data"]["views"]
        key = (metadata["dataset"], metadata["scene"])
        discovered.add(key)

        candidate_count, selected_count = EXPECTED_COUNTS[key]
        assert set(metadata) == {
            "dataset",
            "scene",
            "candidate_count",
            "selected_count",
            "coverage_ratio",
        }
        assert metadata["candidate_count"] == candidate_count
        assert metadata["selected_count"] == selected_count
        assert len(views) == selected_count
        assert len(set(views)) == selected_count
        assert min(views) >= 0
        assert max(views) < candidate_count
        assert 0.0 < metadata["coverage_ratio"] <= 1.0

    assert discovered == set(EXPECTED_COUNTS)


def test_base_config_exposes_optional_view_selection_group() -> None:
    payload = yaml.safe_load((REPO_ROOT / "conf" / "config.yaml").read_text())
    assert {"optional view_selection": None} in payload["defaults"]
