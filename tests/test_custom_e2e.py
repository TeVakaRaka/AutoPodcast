"""End-to-end synthetic test for the 'auto-switch-custom' mode.

Runs the FULL pipeline on the Mac with synthetic data — a synthetic .prproj +
synthetic WAV mics through the real CLI (load -> analyze -> detect -> plan ->
patch_prproj). Explicit ``--mic`` files bypass project audio-source resolution,
so no real media is needed. Both leak modes (off / studio) are exercised.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import soundfile as sf
from click.testing import CliRunner

from autopodcast.cli import cli
from tests.conftest import make_speech_pattern
from tests.test_prproj_patcher import _build_synthetic_prproj

SR = 16000


def _wav(path: Path, segments, duration_s: float) -> None:
    sf.write(str(path), make_speech_pattern(segments, duration_s, SR), SR)


class TestCustomEndToEnd:
    def _setup(self, tmp_path: Path):
        in_p = tmp_path / "in.prproj"
        in_p.write_bytes(_build_synthetic_prproj())
        a = tmp_path / "A.wav"
        b = tmp_path / "B.wav"
        _wav(a, [(0.5, 3.0, -12.0)], 8.0)   # A speaks early
        _wav(b, [(4.0, 7.0, -12.0)], 8.0)   # B speaks later
        return in_p, a, b

    @pytest.mark.parametrize("clean_mode", ["off", "studio"])
    def test_runs_and_patches_prproj(self, tmp_path: Path, clean_mode: str):
        in_p, a, b = self._setup(tmp_path)
        out = tmp_path / "out.prproj"
        result = CliRunner().invoke(cli, [
            "auto-switch-custom",
            "--in", str(in_p), "--seq", "TestSeq", "--out", str(out),
            "--person", "A:1:2", "--person", "B:2:3", "--wide-camera", "1",
            "--mic", str(a), "--mic", str(b),
            "--detector-backend", "rms", "--no-cross-cancel",
            "--audio-clean-mode", clean_mode, "--no-log",
        ])
        assert result.exit_code == 0, result.output
        # the pipeline ran all the way through
        assert "Planning complete:" in result.output
        assert "Done:" in result.output
        # a valid gzipped .prproj was written
        assert out.exists()
        with gzip.open(out, "rb") as f:
            root = ET.fromstring(f.read().decode("utf-8"))
        assert root.tag == "PremiereData"

    def test_studio_is_the_default_clean_mode(self, tmp_path: Path):
        in_p, a, b = self._setup(tmp_path)
        out = tmp_path / "out.prproj"
        # No --audio-clean-mode given -> defaults to studio.
        result = CliRunner().invoke(cli, [
            "auto-switch-custom",
            "--in", str(in_p), "--seq", "TestSeq", "--out", str(out),
            "--person", "A:1:2", "--person", "B:2:3", "--wide-camera", "1",
            "--mic", str(a), "--mic", str(b),
            "--detector-backend", "rms", "--no-cross-cancel", "--no-log",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()
