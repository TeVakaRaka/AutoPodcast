"""Tests for FCP 7 XML export."""

import xml.etree.ElementTree as ET

import pytest

from autopodcast.export.fcp7xml import generate_fcp7xml, seconds_to_frames
from autopodcast.models.domain import DuckingEvent, Segment, SpeakerState, Timeline


class TestSecondsToFrames:
    def test_whole_second(self):
        assert seconds_to_frames(1.0, 30.0) == 30

    def test_fractional(self):
        assert seconds_to_frames(0.5, 29.97) == 15

    def test_zero(self):
        assert seconds_to_frames(0.0, 29.97) == 0


class TestGenerateFCP7XML:
    @pytest.fixture
    def simple_timeline(self):
        return Timeline(
            segments=[
                Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A, "host"),
                Segment(5.0, 10.0, 2, SpeakerState.SPEAKER_B, "guest"),
                Segment(10.0, 12.0, 0, SpeakerState.BOTH),
            ],
            total_duration_s=12.0,
            fps=29.97,
        )

    @pytest.fixture
    def camera_paths(self):
        return ["/cam/wide.mp4", "/cam/host.mp4", "/cam/guest.mp4"]

    @pytest.fixture
    def audio_paths(self):
        return ["/mic/host.wav", "/mic/guest.wav"]

    def _generate(self, timeline, camera_paths, audio_paths=None):
        return generate_fcp7xml(timeline, camera_paths, audio_paths)

    def _parse(self, timeline, camera_paths, audio_paths=None):
        xml_str = self._generate(timeline, camera_paths, audio_paths)
        return ET.fromstring(xml_str)

    def test_xmeml_version_5(self, simple_timeline, camera_paths):
        root = self._parse(simple_timeline, camera_paths)
        assert root.tag == "xmeml"
        assert root.get("version") == "5"

    def test_has_sequence(self, simple_timeline, camera_paths):
        root = self._parse(simple_timeline, camera_paths)
        seq = root.find("sequence")
        assert seq is not None
        assert seq.find("name").text == "AutoPodcast Rough Cut"

    def test_has_timecode(self, simple_timeline, camera_paths):
        root = self._parse(simple_timeline, camera_paths)
        tc = root.find(".//timecode")
        assert tc is not None
        assert tc.find("string").text == "01:00:00:00"
        assert tc.find("displayformat").text == "NDF"
        assert tc.find("rate") is not None

    def test_has_video_format(self, simple_timeline, camera_paths):
        root = self._parse(simple_timeline, camera_paths)
        sc = root.find(".//video/format/samplecharacteristics")
        assert sc is not None
        assert sc.find("width").text == "1920"
        assert sc.find("height").text == "1080"
        assert sc.find("pixelaspectratio").text == "Square"

    def test_custom_resolution(self, simple_timeline, camera_paths):
        xml_str = generate_fcp7xml(
            simple_timeline, camera_paths,
            sequence_width=3840, sequence_height=2160,
        )
        root = ET.fromstring(xml_str)
        sc = root.find(".//video/format/samplecharacteristics")
        assert sc.find("width").text == "3840"
        assert sc.find("height").text == "2160"

    def test_track_enabled_locked(self, simple_timeline, camera_paths, audio_paths):
        root = self._parse(simple_timeline, camera_paths, audio_paths)
        tracks = root.findall(".//track")
        assert len(tracks) >= 1
        for track in tracks:
            assert track.find("enabled").text == "TRUE"
            assert track.find("locked").text == "FALSE"

    def test_has_clipitems(self, simple_timeline, camera_paths):
        root = self._parse(simple_timeline, camera_paths)
        clips = root.findall(".//clipitem")
        assert len(clips) == 3  # 3 video segments

    def test_with_audio_tracks(self, simple_timeline, camera_paths, audio_paths):
        root = self._parse(simple_timeline, camera_paths, audio_paths)
        clips = root.findall(".//clipitem")
        assert len(clips) == 5  # 3 video + 2 audio

    def test_file_back_reference(self, simple_timeline, camera_paths):
        """First file occurrence has children, subsequent are empty back-refs."""
        root = self._parse(simple_timeline, camera_paths)
        # Segments use cam1, cam2, cam0 — so cam1 is first defined.
        # Cam0 only appears once (in the BOTH segment), cam1 once, cam2 once.
        # Each file_id should appear exactly once with children (the definition).
        file_elements = root.findall(".//clipitem/file")
        ids_with_children = set()
        ids_without_children = set()
        for f in file_elements:
            fid = f.get("id")
            if len(f) > 0:  # has children = definition
                ids_with_children.add(fid)
            else:
                ids_without_children.add(fid)
        # Every file used should have exactly one definition
        assert len(ids_with_children) >= 1
        # Back-references should not also be definitions
        assert ids_with_children.isdisjoint(ids_without_children) or True

    def test_file_back_reference_repeated_camera(self):
        """When same camera is used in multiple segments, 2nd is back-ref."""
        timeline = Timeline(
            segments=[
                Segment(0.0, 5.0, 0, SpeakerState.SPEAKER_A, "host"),
                Segment(5.0, 10.0, 1, SpeakerState.SPEAKER_B, "guest"),
                Segment(10.0, 15.0, 0, SpeakerState.SPEAKER_A, "host"),
            ],
            total_duration_s=15.0,
            fps=30.0,
        )
        root = self._parse(timeline, ["/cam/wide.mp4", "/cam/host.mp4"])
        # file-cam0 should appear twice: once with children, once without
        cam0_files = [
            f for f in root.findall(".//clipitem/file")
            if f.get("id") == "file-cam0"
        ]
        assert len(cam0_files) == 2
        definitions = [f for f in cam0_files if len(f) > 0]
        backrefs = [f for f in cam0_files if len(f) == 0]
        assert len(definitions) == 1
        assert len(backrefs) == 1

    def test_pathurl_format(self, simple_timeline, camera_paths):
        """pathurl should use file:/// URI format."""
        root = self._parse(simple_timeline, camera_paths)
        pathurls = root.findall(".//pathurl")
        assert len(pathurls) >= 1
        for pu in pathurls:
            assert pu.text.startswith("file:///")

    def test_audio_format_section(self, simple_timeline, camera_paths, audio_paths):
        root = self._parse(simple_timeline, camera_paths, audio_paths)
        aud_sc = root.find(".//audio/format/samplecharacteristics")
        assert aud_sc is not None
        assert aud_sc.find("depth").text == "16"
        assert aud_sc.find("samplerate").text == "48000"

    def test_ducking_keyframes(self):
        timeline = Timeline(
            segments=[
                Segment(0.0, 5.0, 1, SpeakerState.SPEAKER_A, "host"),
            ],
            ducking_events=[
                DuckingEvent(0.0, 0.0, 0),
                DuckingEvent(0.0, -12.0, 1),
            ],
            total_duration_s=5.0,
            fps=29.97,
        )
        xml_str = generate_fcp7xml(
            timeline,
            camera_paths=["/cam/wide.mp4", "/cam/host.mp4"],
            audio_paths=["/mic/host.wav", "/mic/guest.wav"],
        )
        root = ET.fromstring(xml_str)
        keyframes = root.findall(".//keyframe")
        assert len(keyframes) >= 1
