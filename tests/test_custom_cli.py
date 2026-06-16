"""CLI tests for the configurable 'auto-switch-custom' mode.

These cover the genuinely new surface — ``--person`` parsing and the
pre-flight validation — without needing a real multicam .prproj. The full
analyze->plan->patch chain is covered by the planner unit tests and the
shared patcher tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from autopodcast.cli import _parse_person_spec, cli


class TestParsePersonSpec:
    def test_valid(self):
        assert _parse_person_spec("ведущий:1:2") == ("ведущий", 1, 2)

    def test_strips_whitespace_in_label(self):
        assert _parse_person_spec("  гость 1 :2:3") == ("гость 1", 2, 3)

    @pytest.mark.parametrize("bad", ["host:1", "host:1:2:3", "host", ""])
    def test_wrong_field_count_raises(self, bad):
        with pytest.raises(Exception):
            _parse_person_spec(bad)

    def test_empty_label_raises(self):
        with pytest.raises(Exception):
            _parse_person_spec(":1:2")

    def test_non_integer_raises(self):
        with pytest.raises(Exception):
            _parse_person_spec("host:a:2")

    def test_zero_is_rejected_as_not_one_based(self):
        with pytest.raises(Exception):
            _parse_person_spec("host:0:2")


def _dummy_prproj(tmp_path: Path) -> Path:
    """A placeholder --in file. The validation paths below fail before it's read."""
    p = tmp_path / "in.prproj"
    p.write_bytes(b"dummy")
    return p


def _all_output(result) -> str:
    text = result.output or ""
    try:
        text += result.stderr or ""
    except Exception:  # noqa: BLE001 - stderr may be merged into output
        pass
    return text


class TestCustomCliValidation:
    def _base(self, tmp_path: Path) -> list[str]:
        return [
            "auto-switch-custom",
            "--in", str(_dummy_prproj(tmp_path)),
            "--seq", "Seq",
            "--out", str(tmp_path / "out.prproj"),
        ]

    def test_bad_person_format(self, tmp_path: Path):
        result = CliRunner().invoke(cli, self._base(tmp_path) + [
            "--person", "host:1", "--wide-camera", "1",
        ])
        assert result.exit_code != 0
        assert "camera_angle" in _all_output(result)

    def test_duplicate_tracks_rejected(self, tmp_path: Path):
        result = CliRunner().invoke(cli, self._base(tmp_path) + [
            "--person", "a:1:2", "--person", "b:1:3", "--wide-camera", "1",
        ])
        assert result.exit_code != 0
        assert "unique" in _all_output(result).lower()

    def test_mic_count_mismatch_rejected(self, tmp_path: Path):
        mic = tmp_path / "m.wav"
        mic.write_bytes(b"x")
        result = CliRunner().invoke(cli, self._base(tmp_path) + [
            "--person", "a:1:2", "--person", "b:2:3",
            "--wide-camera", "1", "--mic", str(mic),
        ])
        assert result.exit_code != 0
        assert "--mic" in _all_output(result)

    def test_wide_camera_must_be_one_based(self, tmp_path: Path):
        result = CliRunner().invoke(cli, self._base(tmp_path) + [
            "--person", "a:1:2", "--wide-camera", "0",
        ])
        assert result.exit_code != 0
        assert "1-based" in _all_output(result)

    def test_person_is_required(self, tmp_path: Path):
        result = CliRunner().invoke(cli, self._base(tmp_path) + ["--wide-camera", "1"])
        assert result.exit_code != 0
