"""Tests for speaker role normalization."""

from pathlib import Path

import pytest

from autopodcast.core.roles import normalize_speaker_role
from autopodcast.config import build_config


class TestNormalizeSpeakerRole:
    def test_normalize_host_en(self):
        assert normalize_speaker_role("Host") == "host"
        assert normalize_speaker_role("HOST") == "host"
        assert normalize_speaker_role("presenter") == "host"
        assert normalize_speaker_role("anchor") == "host"

    def test_normalize_guest_en(self):
        assert normalize_speaker_role("Guest") == "guest"
        assert normalize_speaker_role("GUEST") == "guest"
        assert normalize_speaker_role("interviewee") == "guest"
        assert normalize_speaker_role("participant") == "guest"

    def test_normalize_russian(self):
        assert normalize_speaker_role("ведущий") == "host"
        assert normalize_speaker_role("ведущая") == "host"
        assert normalize_speaker_role("гость") == "guest"
        assert normalize_speaker_role("гостья") == "guest"

    def test_unknown_passthrough(self):
        assert normalize_speaker_role("custom") == "custom"
        assert normalize_speaker_role("narrator") == "narrator"

    def test_strips_whitespace(self):
        assert normalize_speaker_role("  host  ") == "host"


class TestRoleOverridePriority:
    def test_override_priority(self):
        """Override dict takes priority over normalize."""
        config = build_config(
            audio_a="a.wav", label_a="presenter", camera_a=1,
            audio_b="b.wav", label_b="guest", camera_b=2,
            speaker_role_overrides={"presenter": "commentator"},
        )
        assert config.audio_inputs[0].speaker_label == "commentator"
        assert config.audio_inputs[1].speaker_label == "guest"

    def test_build_config_normalizes(self):
        """build_config normalizes Russian labels."""
        config = build_config(
            audio_a="a.wav", label_a="ведущий", camera_a=1,
            audio_b="b.wav", label_b="гость", camera_b=2,
        )
        assert config.audio_inputs[0].speaker_label == "host"
        assert config.audio_inputs[1].speaker_label == "guest"
