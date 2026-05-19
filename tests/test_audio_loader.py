"""Tests for audio_loader utilities."""

import numpy as np
import pytest

from autopodcast.core.audio_loader import apply_offset


class TestApplyOffset:
    def test_zero_offset_returns_same(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_offset(audio, 0.0, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_negative_offset_returns_same(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_offset(audio, -1.0, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_trims_beginning(self):
        sr = 16000
        audio = np.arange(sr * 2, dtype=np.float64)  # 2 seconds
        result = apply_offset(audio, 0.5, sr)
        expected_skip = int(0.5 * sr)
        assert len(result) == len(audio) - expected_skip
        np.testing.assert_array_equal(result, audio[expected_skip:])

    def test_offset_beyond_length_returns_empty(self):
        audio = np.array([1.0, 2.0, 3.0])
        result = apply_offset(audio, 10.0, 16000)
        assert len(result) == 0

    def test_preserves_dtype(self):
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_offset(audio, 10.0, 16000)
        assert result.dtype == np.float32
