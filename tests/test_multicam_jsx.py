"""Tests for ExtendScript multicam generator."""

from autopodcast.export.multicam_jsx import generate_multicam_jsx
from autopodcast.models.domain import Segment, SpeakerState, Timeline


def _make_timeline():
    return Timeline(
        segments=[
            Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A, "host"),
            Segment(5.0, 10.0, 2, SpeakerState.SPEAKER_B, "guest"),
            Segment(10.0, 15.0, 0, SpeakerState.BOTH),
        ],
        total_duration_s=15.0,
        fps=29.97,
    )


class TestMulticamJSX:
    def test_jsx_contains_xml_path(self):
        jsx = generate_multicam_jsx("/output/roughcut.xml", _make_timeline(), "Test Seq")
        assert "/output/roughcut.xml" in jsx

    def test_jsx_balanced_braces(self):
        jsx = generate_multicam_jsx("/output/roughcut.xml", _make_timeline(), "Test Seq")
        assert jsx.count("{") == jsx.count("}")
        assert jsx.count("(") == jsx.count(")")

    def test_jsx_has_angle_switches(self):
        jsx = generate_multicam_jsx("/output/roughcut.xml", _make_timeline(), "Test Seq")
        assert "angleSwitches" in jsx
        # Should contain camera indices from segments
        assert '"camera": 1' in jsx
        assert '"camera": 2' in jsx
        assert '"camera": 0' in jsx

    def test_jsx_escapes_special_chars(self):
        jsx = generate_multicam_jsx(
            '/path/with "quotes"/file.xml',
            _make_timeline(),
            'Seq "Name"',
        )
        assert '\\"quotes\\"' in jsx
        assert '\\"Name\\"' in jsx
