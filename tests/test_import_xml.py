"""Tests for Premiere XML import parsing."""

import xml.etree.ElementTree as ET

from autopodcast.import_xml import _parse_rate, _extract_track_clip


class TestParseRate:
    def test_ntsc_23976(self):
        """timebase=24 + ntsc=TRUE → fps ≈ 23.976."""
        rate_xml = "<rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>"
        rate_el = ET.fromstring(rate_xml)
        fps = _parse_rate(rate_el)
        assert abs(fps - 23.976023976) < 0.001

    def test_plain_25fps(self):
        """timebase=25 + ntsc=FALSE → fps = 25.0."""
        rate_xml = "<rate><timebase>25</timebase><ntsc>FALSE</ntsc></rate>"
        rate_el = ET.fromstring(rate_xml)
        fps = _parse_rate(rate_el)
        assert fps == 25.0

    def test_ntsc_2997(self):
        """timebase=30 + ntsc=TRUE → fps ≈ 29.97."""
        rate_xml = "<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>"
        rate_el = ET.fromstring(rate_xml)
        fps = _parse_rate(rate_el)
        assert abs(fps - 29.97002997) < 0.001

    def test_none_returns_default(self):
        fps = _parse_rate(None)
        assert fps == 25.0


class TestExtractTrackClip:
    def test_extract_track_clip_ntsc(self):
        """TrackClip.fps should be NTSC-corrected when ntsc=TRUE."""
        track_xml = """
        <track>
          <clipitem>
            <file>
              <name>clip.mp4</name>
              <pathurl>file:///Volumes/T7/clip.mp4</pathurl>
              <rate>
                <timebase>24</timebase>
                <ntsc>TRUE</ntsc>
              </rate>
            </file>
          </clipitem>
        </track>
        """
        track_el = ET.fromstring(track_xml)
        clip = _extract_track_clip(track_el)
        assert clip is not None
        assert abs(clip.fps - 23.976023976) < 0.001

    def test_extract_track_clip_25fps(self):
        """TrackClip.fps should be 25.0 for plain 25fps."""
        track_xml = """
        <track>
          <clipitem>
            <file>
              <name>clip.mp4</name>
              <pathurl>file:///Volumes/T7/clip.mp4</pathurl>
              <rate>
                <timebase>25</timebase>
                <ntsc>FALSE</ntsc>
              </rate>
            </file>
          </clipitem>
        </track>
        """
        track_el = ET.fromstring(track_xml)
        clip = _extract_track_clip(track_el)
        assert clip is not None
        assert clip.fps == 25.0

    def test_no_clipitem_returns_none(self):
        track_xml = "<track></track>"
        track_el = ET.fromstring(track_xml)
        assert _extract_track_clip(track_el) is None

    def test_extract_track_clip_reads_in_offset(self):
        """TrackClip.source_in_frames should be read from <in> element."""
        track_xml = """
        <track>
          <clipitem>
            <in>156</in>
            <out>54391</out>
            <file>
              <name>C7332.MP4</name>
              <pathurl>file:///Volumes/T7/C7332.MP4</pathurl>
              <rate>
                <timebase>24</timebase>
                <ntsc>TRUE</ntsc>
              </rate>
            </file>
          </clipitem>
        </track>
        """
        track_el = ET.fromstring(track_xml)
        clip = _extract_track_clip(track_el)
        assert clip is not None
        assert clip.source_in_frames == 156

    def test_extract_track_clip_zero_offset(self):
        """TrackClip.source_in_frames should be 0 when <in>0</in>."""
        track_xml = """
        <track>
          <clipitem>
            <in>0</in>
            <out>54265</out>
            <file>
              <name>C9794.MP4</name>
              <pathurl>file:///Volumes/T7/C9794.MP4</pathurl>
              <rate>
                <timebase>24</timebase>
                <ntsc>TRUE</ntsc>
              </rate>
            </file>
          </clipitem>
        </track>
        """
        track_el = ET.fromstring(track_xml)
        clip = _extract_track_clip(track_el)
        assert clip is not None
        assert clip.source_in_frames == 0

    def test_extract_track_clip_missing_in(self):
        """TrackClip.source_in_frames defaults to 0 when no <in> element."""
        track_xml = """
        <track>
          <clipitem>
            <file>
              <name>clip.mp4</name>
              <pathurl>file:///Volumes/T7/clip.mp4</pathurl>
              <rate>
                <timebase>25</timebase>
                <ntsc>FALSE</ntsc>
              </rate>
            </file>
          </clipitem>
        </track>
        """
        track_el = ET.fromstring(track_xml)
        clip = _extract_track_clip(track_el)
        assert clip is not None
        assert clip.source_in_frames == 0
