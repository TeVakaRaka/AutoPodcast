"""Tests for resolving audio sources from Premiere project/XML files."""

from __future__ import annotations

import gzip
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from autopodcast.core.audio_sources import (
    discover_sequence_audio_sources,
    resolve_sequence_audio_sources,
)
from tests.test_prproj_patcher import _build_synthetic_prproj


def _synthetic_prproj_with_audio_paths(host_path: Path, guest_path: Path) -> bytes:
    root = ET.fromstring(gzip.decompress(_build_synthetic_prproj()).decode("utf-8"))
    for subclip in root.iter("SubClip"):
        name_el = subclip.find("Name")
        if name_el is None or not name_el.text:
            continue
        if name_el.text == "host_mic.wav":
            ET.SubElement(subclip, "ActualMediaFilePath").text = str(host_path)
        elif name_el.text == "guest_mic.wav":
            ET.SubElement(subclip, "ActualMediaFilePath").text = str(guest_path)
    return gzip.compress(ET.tostring(root, encoding="utf-8"))


def test_discovers_audio_sources_from_prproj(tmp_path: Path):
    host = tmp_path / "host.wav"
    guest = tmp_path / "guest.wav"
    prproj = tmp_path / "episode.prproj"
    prproj.write_bytes(_synthetic_prproj_with_audio_paths(host, guest))

    resolved = discover_sequence_audio_sources(prproj, "TestSeq")

    assert resolved.method == "prproj"
    assert resolved.sources_by_track[0].representative_path == host
    assert resolved.sources_by_track[1].representative_path == guest


def test_resolve_selected_audio_tracks_reports_missing_tracks(tmp_path: Path):
    prproj = tmp_path / "episode.prproj"
    prproj.write_bytes(
        _synthetic_prproj_with_audio_paths(tmp_path / "host.wav", tmp_path / "guest.wav")
    )

    with pytest.raises(ValueError, match="track\\(s\\): 3"):
        resolve_sequence_audio_sources(prproj, "TestSeq", [0, 2])


def test_discovers_audio_sources_from_xml(tmp_path: Path):
    main = tmp_path / "main.wav"
    cohost = tmp_path / "cohost.wav"
    guest = tmp_path / "guest.wav"
    xml_path = tmp_path / "episode.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>Seq</name>
    <media>
      <audio>
        <track>
          <clipitem><file><name>main</name><pathurl>file://localhost{main.as_posix()}</pathurl></file></clipitem>
        </track>
        <track>
          <clipitem><file><name>cohost</name><pathurl>file://localhost{cohost.as_posix()}</pathurl></file></clipitem>
        </track>
        <track>
          <clipitem><file><name>guest</name><pathurl>file://localhost{guest.as_posix()}</pathurl></file></clipitem>
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>
""",
        encoding="utf-8",
    )
    prproj = tmp_path / "unused.prproj"
    prproj.write_bytes(b"unused")

    resolved = resolve_sequence_audio_sources(prproj, "Seq", [0, 1, 2], xml_path=str(xml_path))

    assert resolved.method == "xml"
    assert resolved.sources_by_track[0].representative_path == main
    assert resolved.sources_by_track[1].representative_path == cohost
    assert resolved.sources_by_track[2].representative_path == guest


def test_discovers_audio_sources_from_named_xml_sequence(tmp_path: Path):
    wrong = tmp_path / "wrong.wav"
    right = tmp_path / "right.wav"
    xml_path = tmp_path / "episode.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>OtherSeq</name>
    <media><audio><track>
      <clipitem><file><name>wrong</name><pathurl>file://localhost{wrong.as_posix()}</pathurl></file></clipitem>
    </track></audio></media>
  </sequence>
  <sequence>
    <name>TargetSeq</name>
    <media><audio><track>
      <clipitem><file><name>right</name><pathurl>file://localhost{right.as_posix()}</pathurl></file></clipitem>
    </track></audio></media>
  </sequence>
</xmeml>
""",
        encoding="utf-8",
    )
    prproj = tmp_path / "unused.prproj"
    prproj.write_bytes(b"unused")

    resolved = resolve_sequence_audio_sources(
        prproj,
        "TargetSeq",
        [0],
        xml_path=str(xml_path),
    )

    assert resolved.sources_by_track[0].representative_path == right
