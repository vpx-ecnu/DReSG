from __future__ import annotations

import json
from pathlib import Path

import pytest

import dresg.utils.json_io as json_io
from dresg.utils.json_io import load_json, save_json


def test_save_json_atomically_replaces_the_target(tmp_path) -> None:
    path = tmp_path / "nested" / "metrics.json"
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.parent.mkdir(parents=True)
    temporary_path.write_text("stale", encoding="utf-8")

    save_json(path, {"status": "running", "step": 3})

    assert load_json(path) == {"status": "running", "step": 3}
    assert not temporary_path.exists()


def test_save_json_preserves_the_target_when_replace_fails(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "summary.json"
    old_payload = {"status": "running", "step": 2}
    path.write_text(json.dumps(old_payload), encoding="utf-8")
    temporary_path = path.with_name(f".{path.name}.tmp")

    def fail_replace(source: Path, target: Path) -> None:
        assert Path(source) == temporary_path
        assert Path(target) == path
        raise OSError("replace failed")

    monkeypatch.setattr(json_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_json(path, {"status": "running", "step": 3})

    assert load_json(path) == old_payload
    assert not temporary_path.exists()


def test_save_json_preserves_the_target_on_serialization_error(tmp_path) -> None:
    path = tmp_path / "summary.json"
    old_payload = {"status": "running"}
    path.write_text(json.dumps(old_payload), encoding="utf-8")

    with pytest.raises(TypeError):
        save_json(path, {"unsupported": object()})

    assert load_json(path) == old_payload
    assert not path.with_name(f".{path.name}.tmp").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_save_json_rejects_nonfinite_numbers_without_touching_target(
    tmp_path,
    value: float,
) -> None:
    path = tmp_path / "summary.json"
    old_payload = {"status": "running"}
    path.write_text(json.dumps(old_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON compliant"):
        save_json(path, {"metric": value})

    assert load_json(path) == old_payload
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_load_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"step": 1, "step": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeats key"):
        load_json(path)

    path.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_json(path)
