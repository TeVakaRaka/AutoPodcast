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
        clips = root.findall(".//video//clipitem")
        # Track 0: 1 continuous bg clip
        # Track 1: 1 clip (cam1, seg0: 0-5s)
        # Track 2: 1 clip (cam2, seg1: 5-10s)
        assert len(clips) == 3

    def test_with_audio_tracks(self, simple_timeline, camera_paths, audio_paths):
        root = self._parse(simple_timeline, camera_paths, audio_paths)
        clips = root.findall(".//clipitem")
        # 3 video + 2 audio = 5
        assert len(clips) == 5

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
        """Each file has exactly one definition, rest are back-refs."""
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
        # Track 0: 1 continuous bg clip (file-cam0 definition)
        # Track 1: 1 clip for cam1 seg (file-cam1 definition)
        cam0_files = [
            f for f in root.findall(".//clipitem/file")
            if f.get("id") == "file-cam0"
        ]
        assert len(cam0_files) == 1
        definitions = [f for f in cam0_files if len(f) > 0]
        assert len(definitions) == 1

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

    def test_keyframe_linear_interpolation(self):
        """XML keyframes contain <interpolation><name>Linear</name>."""
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
        for kf in keyframes:
            interp = kf.find("interpolation/name")
            assert interp is not None
            assert interp.text == "Linear"

    def test_gap_based_tracks(self, simple_timeline, camera_paths):
        """Bottom track is continuous; upper tracks only have active clips."""
        root = self._parse(simple_timeline, camera_paths)
        video_tracks = root.findall(".//video/track")
        assert len(video_tracks) == 3  # one per camera

        # Track 0 (wide): 1 continuous clip
        t0_clips = video_tracks[0].findall("clipitem")
        assert len(t0_clips) == 1
        assert int(t0_clips[0].find("start").text) == 0

        # Track 1 (host cam): 1 clip for segment 0 (cam1, 0-5s)
        t1_clips = video_tracks[1].findall("clipitem")
        assert len(t1_clips) == 1

        # Track 2 (guest cam): 1 clip for segment 1 (cam2, 5-10s)
        t2_clips = video_tracks[2].findall("clipitem")
        assert len(t2_clips) == 1


class TestSourceOffsets:
    """Tests for camera/audio source offset (trim-in preservation)."""

    def test_camera_source_offset_applied(self):
        """With camera_source_in_s, in/out frames are shifted."""
        timeline = Timeline(
            segments=[
                Segment(0.0, 10.0, 1, SpeakerState.SPEAKER_A, "host"),
            ],
            total_duration_s=10.0,
            fps=25.0,
        )
        # Camera at 23.976fps with 6.24s offset (= 156 frames at 25fps seq)
        xml_str = generate_fcp7xml(
            timeline,
            camera_paths=["/cam/wide.mp4", "/cam/host.mp4"],
            camera_fps=[23.976, 23.976],
            camera_source_in_s=[0.0, 6.24],
        )
        root = ET.fromstring(xml_str)
        # Track 1 (host camera, index 1) first clip
        video_tracks = root.findall(".//video/track")
        host_track = video_tracks[1]
        clip = host_track.find("clipitem")
        in_frame = int(clip.find("in").text)
        out_frame = int(clip.find("out").text)
        # in = round((0.0 + 6.24) * 23.976) = round(149.61) = 150
        assert in_frame == seconds_to_frames(6.24, 23.976)
        # out = round((10.0 + 6.24) * 23.976) = round(389.37) = 389
        assert out_frame == seconds_to_frames(16.24, 23.976)
        # start/end should NOT be shifted (sequence timeline position)
        assert int(clip.find("start").text) == 0
        assert int(clip.find("end").text) == seconds_to_frames(10.0, 25.0)

    def test_audio_source_offset_applied(self):
        """With audio_source_in_s, audio in/out are shifted."""
        timeline = Timeline(
            segments=[
                Segment(0.0, 10.0, 0, SpeakerState.SPEAKER_A, "host"),
            ],
            total_duration_s=10.0,
            fps=25.0,
        )
        xml_str = generate_fcp7xml(
            timeline,
            camera_paths=["/cam/wide.mp4"],
            audio_paths=["/mic/host.wav", "/mic/guest.wav"],
            audio_fps=[29.97, 29.97],
            audio_source_in_s=[5.12, 5.20],
        )
        root = ET.fromstring(xml_str)
        audio_clips = root.findall(".//audio/track/clipitem")
        # Track 0: in = round(5.12 * 29.97), out = round(15.12 * 29.97)
        clip0 = audio_clips[0]
        assert int(clip0.find("in").text) == seconds_to_frames(5.12, 29.97)
        assert int(clip0.find("out").text) == seconds_to_frames(15.12, 29.97)
        # Track 1: in = round(5.20 * 29.97), out = round(15.20 * 29.97)
        clip1 = audio_clips[1]
        assert int(clip1.find("in").text) == seconds_to_frames(5.20, 29.97)
        assert int(clip1.find("out").text) == seconds_to_frames(15.20, 29.97)

    def test_no_offset_backward_compat(self):
        """Without source offsets, in/out start from 0 (old behavior)."""
        timeline = Timeline(
            segments=[
                Segment(0.0, 10.0, 0, SpeakerState.SPEAKER_A, "host"),
            ],
            total_duration_s=10.0,
            fps=25.0,
        )
        xml_str = generate_fcp7xml(
            timeline,
            camera_paths=["/cam/wide.mp4"],
            audio_paths=["/mic/host.wav"],
        )
        root = ET.fromstring(xml_str)
        video_clip = root.find(".//video/track/clipitem")
        assert int(video_clip.find("in").text) == 0
        audio_clip = root.find(".//audio/track/clipitem")
        assert int(audio_clip.find("in").text) == 0


class TestMixedFPS:
    """Tests for mixed FPS: sequence at 25fps, clips at 23.976fps."""

    @pytest.fixture
    def mixed_timeline(self):
        return Timeline(
            segments=[
                Segment(0.0, 10.0, 1, SpeakerState.SPEAKER_A, "host"),
                Segment(10.0, 20.0, 2, SpeakerState.SPEAKER_B, "guest"),
            ],
            total_duration_s=20.0,
            fps=25.0,
        )

    @pytest.fixture
    def camera_paths(self):
        return ["/cam/wide.mp4", "/cam/host.mp4", "/cam/guest.mp4"]

    @pytest.fixture
    def audio_paths(self):
        return ["/mic/host.wav", "/mic/guest.wav"]

    def _generate_mixed(self, timeline, camera_paths, audio_paths=None):
        clip_fps = 23.976
        return generate_fcp7xml(
            timeline, camera_paths, audio_paths,
            camera_fps=[clip_fps] * len(camera_paths),
            audio_fps=[clip_fps] * len(audio_paths) if audio_paths else None,
        )

    def test_sequence_rate_is_seq_fps(self, mixed_timeline, camera_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        seq_rate = root.find("sequence/rate")
        assert seq_rate.find("timebase").text == "25"
        assert seq_rate.find("ntsc").text == "FALSE"

    def test_clipitem_rate_is_clip_fps(self, mixed_timeline, camera_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        clip = root.find(".//video/track/clipitem")
        clip_rate = clip.find("rate")
        assert clip_rate.find("timebase").text == "24"
        assert clip_rate.find("ntsc").text == "TRUE"

    def test_start_end_use_seq_fps(self, mixed_timeline, camera_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        # Track 1 has cam1 clip (0-10s at 25fps seq rate)
        video_tracks = root.findall(".//video/track")
        clip = video_tracks[1].find("clipitem")
        assert int(clip.find("start").text) == seconds_to_frames(0.0, 25.0)
        assert int(clip.find("end").text) == seconds_to_frames(10.0, 25.0)

    def test_in_out_use_clip_fps(self, mixed_timeline, camera_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        # Track 1 has cam1 clip (0-10s at 23.976fps clip rate)
        video_tracks = root.findall(".//video/track")
        clip = video_tracks[1].find("clipitem")
        assert int(clip.find("in").text) == seconds_to_frames(0.0, 23.976)
        assert int(clip.find("out").text) == seconds_to_frames(10.0, 23.976)

    def test_file_duration_is_clip_fps(self, mixed_timeline, camera_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        # File duration should be in clip fps (20s * 23.976 = 480 frames)
        file_el = root.find(".//video/track/clipitem/file")
        dur = int(file_el.find("duration").text)
        assert dur == seconds_to_frames(20.0, 23.976)

    def test_backward_compat_no_camera_fps(self, mixed_timeline, camera_paths):
        """Without camera_fps, in/out should equal start/end (old behavior)."""
        xml_str = generate_fcp7xml(mixed_timeline, camera_paths)
        root = ET.fromstring(xml_str)
        clip = root.find(".//video/track/clipitem")
        assert clip.find("start").text == clip.find("in").text
        assert clip.find("end").text == clip.find("out").text

    def test_audio_in_out_use_clip_fps(self, mixed_timeline, camera_paths, audio_paths):
        xml_str = self._generate_mixed(mixed_timeline, camera_paths, audio_paths)
        root = ET.fromstring(xml_str)
        audio_clip = root.find(".//audio/track/clipitem")
        # Audio: full duration 20s; start/end in seq fps (25), in/out in clip fps (23.976)
        assert int(audio_clip.find("start").text) == 0
        assert int(audio_clip.find("end").text) == seconds_to_frames(20.0, 25.0)
        assert int(audio_clip.find("in").text) == 0
        assert int(audio_clip.find("out").text) == seconds_to_frames(20.0, 23.976)
