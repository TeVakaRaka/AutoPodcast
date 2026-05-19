"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from autopodcast.cli import cli
from tests.conftest import make_speech_pattern, make_silence


@pytest.fixture
def tmp_wav_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two 10s WAV files: host speaks 1-3s, guest speaks 5-8s."""
    sr = 16000
    host = make_speech_pattern([(1.0, 3.0, -12.0)], 10.0, sr)
    guest = make_speech_pattern([(5.0, 8.0, -12.0)], 10.0, sr)
    path_a = tmp_path / "host.wav"
    path_b = tmp_path / "guest.wav"
    sf.write(str(path_a), host, sr)
    sf.write(str(path_b), guest, sr)
    return path_a, path_b


@pytest.fixture
def tmp_silence_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two 5s silent WAV files."""
    sr = 16000
    silence = make_silence(5.0, sr)
    path_a = tmp_path / "silence_a.wav"
    path_b = tmp_path / "silence_b.wav"
    sf.write(str(path_a), silence, sr)
    sf.write(str(path_b), silence, sr)
    return path_a, path_b


class TestAnalyzeCommand:
    def test_basic_run_exits_zero(self, tmp_wav_pair: tuple[Path, Path], tmp_path: Path):
        a, b = tmp_wav_pair
        out = tmp_path / "timeline.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--audio-a", str(a), "--audio-b", str(b),
            "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_contains_segment_count(self, tmp_wav_pair: tuple[Path, Path], tmp_path: Path):
        a, b = tmp_wav_pair
        out = tmp_path / "timeline.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--audio-a", str(a), "--audio-b", str(b),
            "-o", str(out),
        ])
        assert "segment" in result.output.lower()

    def test_invalid_camera_index_rejected(self, tmp_wav_pair: tuple[Path, Path], tmp_path: Path):
        a, b = tmp_wav_pair
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--audio-a", str(a), "--audio-b", str(b),
            "--camera-a", "0",
            "-o", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0

    def test_label_flags_accepted(self, tmp_wav_pair: tuple[Path, Path], tmp_path: Path):
        a, b = tmp_wav_pair
        out = tmp_path / "timeline.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--audio-a", str(a), "--audio-b", str(b),
            "--label-a", "alice", "--label-b", "bob",
            "-o", str(out),
        ])
        assert result.exit_code == 0, result.output

    def test_missing_required_args_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0

    def test_auto_switch_4cams_help_is_available(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["auto-switch-4cams", "--help"])
        assert result.exit_code == 0, result.output
        assert "--mic-host" in result.output
        assert "--camera-guest-close" in result.output
        assert "--motion-check / --no-motion-check" in result.output
        assert "--motion-hwaccel" in result.output


class TestCalibrateCommand:
    def test_silent_mics_low_noise_floor(self, tmp_silence_pair: tuple[Path, Path]):
        a, b = tmp_silence_pair
        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate",
            "--mic-a", str(a), "--mic-b", str(b),
        ])
        assert result.exit_code == 0, result.output
        assert "Noise floor" in result.output

    def test_output_includes_suggested_flags(self, tmp_silence_pair: tuple[Path, Path]):
        a, b = tmp_silence_pair
        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate",
            "--mic-a", str(a), "--mic-b", str(b),
        ])
        assert "--speech-threshold" in result.output

    def test_noisy_mic_raises_threshold(self, tmp_path: Path):
        sr = 16000
        rng = np.random.default_rng(42)
        # Quiet mic
        quiet = rng.normal(0, 0.001, sr * 5)
        # Noisy mic
        noisy = rng.normal(0, 0.03, sr * 5)

        quiet_path = tmp_path / "quiet.wav"
        noisy_path = tmp_path / "noisy.wav"
        sf.write(str(quiet_path), quiet, sr)
        sf.write(str(noisy_path), noisy, sr)

        runner = CliRunner()
        result_quiet = runner.invoke(cli, [
            "calibrate", "--mic-a", str(quiet_path), "--mic-b", str(quiet_path),
        ])
        result_noisy = runner.invoke(cli, [
            "calibrate", "--mic-a", str(noisy_path), "--mic-b", str(noisy_path),
        ])

        assert result_quiet.exit_code == 0
        assert result_noisy.exit_code == 0

        # Extract speech threshold from "Speech threshold:  X.X dB"
        def extract_threshold(output: str) -> float:
            for line in output.splitlines():
                if "Speech threshold:" in line:
                    return float(line.strip().split()[-2])
            raise ValueError("No threshold found")

        quiet_thresh = extract_threshold(result_quiet.output)
        noisy_thresh = extract_threshold(result_noisy.output)
        assert noisy_thresh > quiet_thresh

    def test_missing_mic_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["calibrate"])
        assert result.exit_code != 0


class TestConfigValidation:
    def test_inverted_thresholds_rejected(self, tmp_wav_pair: tuple[Path, Path], tmp_path: Path):
        a, b = tmp_wav_pair
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "--audio-a", str(a), "--audio-b", str(b),
            "--speech-threshold", "-35",
            "--release-threshold", "-30",
            "-o", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "threshold" in str(result.exception).lower()
