"""Tests for prproj_patcher module."""

from __future__ import annotations

import gzip
import json
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from autopodcast.models.domain import Segment, SpeakerState
from autopodcast.prproj_patcher import (
    TICKS_PER_SECOND,
    _build_full_segments,
    _find_sequence,
    _read_sequence_tpf,
    build_segments,
    expand_audio_intervals,
    fps_to_ticks_per_frame,
    patch_prproj,
    read_audio_offsets,
    seconds_to_ticks,
    segments_to_audio_intervals,
    segments_to_cuts,
    snap_ticks_to_frame,
)


class TestSecondsToTicks:
    def test_zero(self):
        assert seconds_to_ticks(0) == 0

    def test_one_second(self):
        assert seconds_to_ticks(1.0) == TICKS_PER_SECOND

    def test_fractional(self):
        result = seconds_to_ticks(502.92)
        assert result == round(502.92 * TICKS_PER_SECOND)

    def test_negative(self):
        assert seconds_to_ticks(-1.0) == -TICKS_PER_SECOND


class TestBuildSegments:
    def test_no_cuts(self):
        segs = build_segments(0, 1000, 0, [])
        assert segs == [(0, 1000, 0)]

    def test_one_cut(self):
        end = seconds_to_ticks(100)
        cut_time = 50.0
        segs = build_segments(0, end, 0, [{"time": cut_time, "angle": 1}])
        assert len(segs) == 2
        assert segs[0] == (0, seconds_to_ticks(50), 0)
        assert segs[1] == (seconds_to_ticks(50), end, 1)

    def test_multiple_cuts(self):
        end = seconds_to_ticks(2000)
        cuts = [
            {"time": 500, "angle": 1},
            {"time": 1000, "angle": 2},
            {"time": 1500, "angle": 0},
        ]
        segs = build_segments(0, end, 0, cuts)
        assert len(segs) == 4
        assert segs[0][2] == 0  # original angle
        assert segs[1][2] == 1
        assert segs[2][2] == 2
        assert segs[3][2] == 0
        # Verify continuity
        for i in range(len(segs) - 1):
            assert segs[i][1] == segs[i + 1][0]

    def test_cut_at_boundaries_ignored(self):
        end = seconds_to_ticks(100)
        cuts = [
            {"time": 0, "angle": 1},    # at start — ignored
            {"time": 50, "angle": 2},    # valid
            {"time": 100, "angle": 0},   # at end — ignored
        ]
        segs = build_segments(0, end, 0, cuts)
        assert len(segs) == 2

    def test_preserves_original_angle(self):
        end = seconds_to_ticks(100)
        segs = build_segments(0, end, 2, [{"time": 50, "angle": 1}])
        assert segs[0][2] == 2  # original angle preserved


class TestFindSequence:
    def test_finds_root_level_sequence_by_name(self):
        root = ET.Element("PremiereData")
        seq = ET.SubElement(root, "Sequence")
        ET.SubElement(seq, "Name").text = "Основа 1"

        assert _find_sequence(root, "Основа 1") is seq

    def test_finds_nested_sequence_by_name(self):
        root = ET.Element("PremiereData")
        wrapper = ET.SubElement(root, "Bin")
        seq = ET.SubElement(wrapper, "Sequence")
        ET.SubElement(seq, "Name").text = "Основа 1"

        assert _find_sequence(root, "Основа 1") is seq

    def test_finds_sequence_by_wrapper_name_and_object_ref(self):
        root = ET.Element("PremiereData")
        seq = ET.SubElement(root, "Sequence", ObjectID="seq-1")
        wrapper = ET.SubElement(root, "ProjectItem")
        ET.SubElement(wrapper, "Name").text = "Основа 1"
        ET.SubElement(wrapper, "Sequence", ObjectRef="seq-1")

        assert _find_sequence(root, "Основа 1") is seq

    def test_normalizes_unicode_spacing(self):
        root = ET.Element("PremiereData")
        seq = ET.SubElement(root, "Sequence")
        ET.SubElement(seq, "Name").text = "Основа\u00a01"

        assert _find_sequence(root, "Основа 1") is seq

    def test_not_found_lists_available_sequences(self):
        root = ET.Element("PremiereData")
        seq = ET.SubElement(root, "Sequence")
        ET.SubElement(seq, "Name").text = "Основа 1"

        with pytest.raises(ValueError, match="Available sequences"):
            _find_sequence(root, "Другая секвенция")


def _seg(start: float, end: float, cam: int, state: SpeakerState = SpeakerState.SPEAKER_A) -> Segment:
    return Segment(start_s=start, end_s=end, camera_index=cam, speaker_state=state)


class TestSegmentsToCuts:
    def test_empty(self):
        first, cuts = segments_to_cuts([])
        assert first == 0
        assert cuts == []

    def test_single_segment(self):
        first, cuts = segments_to_cuts([_seg(0, 100, 1)])
        assert first == 1
        assert cuts == []

    def test_two_segments_different_cameras(self):
        first, cuts = segments_to_cuts([_seg(0, 50, 1), _seg(50, 100, 2)])
        assert first == 1
        assert cuts == [{"time": 50, "angle": 2}]

    def test_consecutive_same_camera_merged(self):
        first, cuts = segments_to_cuts([
            _seg(0, 30, 1),
            _seg(30, 60, 1),  # same camera — no cut
            _seg(60, 100, 2),
        ])
        assert first == 1
        assert cuts == [{"time": 60, "angle": 2}]

    def test_multiple_switches(self):
        first, cuts = segments_to_cuts([
            _seg(0, 10, 0),
            _seg(10, 20, 1),
            _seg(20, 30, 2),
            _seg(30, 40, 0),
        ])
        assert first == 0
        assert len(cuts) == 3
        assert cuts[0] == {"time": 10, "angle": 1}
        assert cuts[1] == {"time": 20, "angle": 2}
        assert cuts[2] == {"time": 30, "angle": 0}


def _build_synthetic_prproj() -> bytes:
    """Build a minimal synthetic .prproj XML for testing."""
    root = ET.Element("PremiereData", Version="3")

    # Project
    proj_ref = ET.SubElement(root, "Project")
    proj_ref.set("ObjectRef", "1")

    proj = ET.SubElement(root, "Project")
    proj.set("ObjectID", "1")
    proj.set("ClassID", "62ad66dd-0dcd-42da-a660-6d8fbde94876")
    proj.set("Version", "42")
    next_id = ET.SubElement(proj, "NextID")
    next_id.text = "1000010"

    # VideoTrackGroup
    vtg = ET.SubElement(root, "VideoTrackGroup")
    vtg.set("ObjectID", "10")
    vtg.set("ClassID", "9e9abf7a-0918-49c2-91ae-991b5dde77bb")
    vtg.set("Version", "13")
    tg_inner = ET.SubElement(vtg, "TrackGroup")
    tg_inner.set("Version", "1")
    fr = ET.SubElement(tg_inner, "FrameRate")
    fr.text = str(10160640000)  # 25fps
    tracks = ET.SubElement(tg_inner, "Tracks")
    tracks.set("Version", "1")
    track_ref = ET.SubElement(tracks, "Track")
    track_ref.set("Index", "0")
    track_ref.set("ObjectURef", "track-uid-001")

    # VideoClipTrack
    vct = ET.SubElement(root, "VideoClipTrack")
    vct.set("ObjectUID", "track-uid-001")
    vct.set("ClassID", "f68dcd81-8805-11d5-af2d-9bfa89d4ddd4")
    vct.set("Version", "1")
    ct = ET.SubElement(vct, "ClipTrack")
    ct.set("Version", "2")
    trk = ET.SubElement(ct, "Track")
    trk.set("Version", "3")
    ci = ET.SubElement(ct, "ClipItems")
    ci.set("Version", "3")
    tis = ET.SubElement(ci, "TrackItems")
    tis.set("Version", "1")
    ti = ET.SubElement(tis, "TrackItem")
    ti.set("Index", "0")
    ti.set("ObjectRef", "20")

    # VideoSequenceSource (the Source object)
    vss = ET.SubElement(root, "VideoSequenceSource")
    vss.set("ObjectID", "50")
    vss.set("ClassID", "dummy-source-class")
    vss.set("Version", "1")

    # VideoClip (the actual clip with multicam data)
    vc = ET.SubElement(root, "VideoClip")
    vc.set("ObjectID", "30")
    vc.set("ClassID", "9308dbef-2440-4acb-9ab2-953b9a4e82ec")
    vc.set("Version", "11")
    clip = ET.SubElement(vc, "Clip")
    clip.set("Version", "18")
    cn = ET.SubElement(clip, "Node")
    cn.set("Version", "1")
    cp = ET.SubElement(cn, "Properties")
    cp.set("Version", "1")
    lbl = ET.SubElement(cp, "asl.clip.label.name")
    lbl.text = "BE.Prefs.LabelColors.5"
    src = ET.SubElement(clip, "Source")
    src.set("ObjectRef", "50")
    cid = ET.SubElement(clip, "ClipID")
    cid.text = str(uuid.uuid4())
    inp = ET.SubElement(clip, "InPoint")
    inp.text = "0"
    outp = ET.SubElement(clip, "OutPoint")
    outp.text = str(seconds_to_ticks(100))
    imc = ET.SubElement(clip, "IsMulticam")
    imc.text = "true"
    sti = ET.SubElement(clip, "SelectedTrackIndex")
    sti.text = "0"

    # SubClip
    sc = ET.SubElement(root, "SubClip")
    sc.set("ObjectID", "25")
    sc.set("ClassID", "e0c58dc9-dbdd-4166-aef7-5db7e3f22e84")
    sc.set("Version", "5")
    sc_clip = ET.SubElement(sc, "Clip")
    sc_clip.set("ObjectRef", "30")
    mc = ET.SubElement(sc, "MasterClip")
    mc.set("ObjectURef", str(uuid.uuid4()))
    nm = ET.SubElement(sc, "Name")
    nm.text = "Многокам"
    ocg = ET.SubElement(sc, "OrigChGrp")
    ocg.text = "0"

    # VideoComponentChain
    vcc = ET.SubElement(root, "VideoComponentChain")
    vcc.set("ObjectID", "22")
    vcc.set("ClassID", "0970e08a-f58f-4108-b29a-1a717b8e12e2")
    vcc.set("Version", "3")
    dm = ET.SubElement(vcc, "DefaultMotion")
    dm.text = "true"
    do = ET.SubElement(vcc, "DefaultOpacity")
    do.text = "true"
    dmc = ET.SubElement(vcc, "DefaultMotionComponentID")
    dmc.text = "1"
    doc = ET.SubElement(vcc, "DefaultOpacityComponentID")
    doc.text = "2"
    cc = ET.SubElement(vcc, "ComponentChain")
    cc.set("Version", "3")
    ccn = ET.SubElement(cc, "Node")
    ccn.set("Version", "1")
    ccp = ET.SubElement(ccn, "Properties")
    ccp.set("Version", "1")
    acid = ET.SubElement(ccp, "MZ.ComponentChain.ActiveComponentID")
    acid.text = "2"
    acpi = ET.SubElement(ccp, "MZ.ComponentChain.ActiveComponentParamIndex")
    acpi.text = "4294967295"

    # VideoClipTrackItem
    vcti = ET.SubElement(root, "VideoClipTrackItem")
    vcti.set("ObjectID", "20")
    vcti.set("ClassID", "368b0406-29e3-4923-9fcd-094fbf9a1089")
    vcti.set("Version", "6")
    cti = ET.SubElement(vcti, "ClipTrackItem")
    cti.set("Version", "8")
    co = ET.SubElement(cti, "ComponentOwner")
    co.set("Version", "1")
    comp = ET.SubElement(co, "Components")
    comp.set("ObjectRef", "22")
    ti_inner = ET.SubElement(cti, "TrackItem")
    ti_inner.set("Version", "3")
    n = ET.SubElement(ti_inner, "Node")
    n.set("Version", "1")
    nid = ET.SubElement(n, "ID")
    nid.text = "1000009"
    s = ET.SubElement(ti_inner, "Start")
    s.text = "0"
    e = ET.SubElement(ti_inner, "End")
    e.text = str(seconds_to_ticks(100))
    sc_ref = ET.SubElement(cti, "SubClip")
    sc_ref.set("ObjectRef", "25")
    fr = ET.SubElement(vcti, "FrameRect")
    fr.text = "0,0,1920,1080"
    pa = ET.SubElement(vcti, "PixelAspectRatio")
    pa.text = "1,1"

    # --- Audio objects ---

    # AudioMediaSource (source for audio clips)
    ams = ET.SubElement(root, "AudioMediaSource")
    ams.set("ObjectID", "60")
    ams.set("ClassID", "dummy-audio-source-class")
    ams.set("Version", "1")

    audio_master_uref = str(uuid.uuid4())

    # Helper to create one audio track with its clip chain
    def _make_audio_track(track_uid, item_oid, comp_oid, subclip_oid, clip_oid, clip_name):
        # AudioComponentChain
        acc = ET.SubElement(root, "AudioComponentChain")
        acc.set("ObjectID", comp_oid)
        acc.set("ClassID", "3cb131d1-d3c0-47ae-a19a-bdf75ea11674")
        acc.set("Version", "3")
        dv = ET.SubElement(acc, "DefaultVol")
        dv.text = "true"
        dvci = ET.SubElement(acc, "DefaultVolumeComponentID")
        dvci.text = "1"
        cc_a = ET.SubElement(acc, "ComponentChain")
        cc_a.set("Version", "3")
        ccn_a = ET.SubElement(cc_a, "Node")
        ccn_a.set("Version", "1")
        ccp_a = ET.SubElement(ccn_a, "Properties")
        ccp_a.set("Version", "1")
        a_acid = ET.SubElement(ccp_a, "MZ.ComponentChain.ActiveComponentID")
        a_acid.text = "1"
        a_acpi = ET.SubElement(ccp_a, "MZ.ComponentChain.ActiveComponentParamIndex")
        a_acpi.text = "4294967295"
        acl = ET.SubElement(acc, "AudioChannelLayout")
        acl.text = '[{"channellabel":0}]'
        ct_a = ET.SubElement(acc, "ChannelType")
        ct_a.text = "0"
        fr_a = ET.SubElement(acc, "FrameRate")
        fr_a.text = "5292000"
        am_a = ET.SubElement(acc, "AutomationMode")
        am_a.text = "1"

        # AudioClip
        a_clip = ET.SubElement(root, "AudioClip")
        a_clip.set("ObjectID", clip_oid)
        a_clip.set("ClassID", "b8830d03-de02-41ee-84ec-fe566dc70cd9")
        a_clip.set("Version", "8")
        a_clip_inner = ET.SubElement(a_clip, "Clip")
        a_clip_inner.set("Version", "18")
        a_cn = ET.SubElement(a_clip_inner, "Node")
        a_cn.set("Version", "1")
        a_cp = ET.SubElement(a_cn, "Properties")
        a_cp.set("Version", "1")
        a_lbl = ET.SubElement(a_cp, "asl.clip.label.name")
        a_lbl.text = "BE.Prefs.LabelColors.2"
        a_src = ET.SubElement(a_clip_inner, "Source")
        a_src.set("ObjectRef", "60")
        a_cid = ET.SubElement(a_clip_inner, "ClipID")
        a_cid.text = str(uuid.uuid4())
        a_inp = ET.SubElement(a_clip_inner, "InPoint")
        a_inp.text = "0"
        a_outp = ET.SubElement(a_clip_inner, "OutPoint")
        a_outp.text = str(seconds_to_ticks(100))
        a_acl = ET.SubElement(a_clip, "AudioChannelLayout")
        a_acl.text = '[{"channellabel":0}]'

        # SubClip for audio
        a_sc = ET.SubElement(root, "SubClip")
        a_sc.set("ObjectID", subclip_oid)
        a_sc.set("ClassID", "e0c58dc9-dbdd-4166-aef7-5db7e3f22e84")
        a_sc.set("Version", "5")
        a_sc_clip = ET.SubElement(a_sc, "Clip")
        a_sc_clip.set("ObjectRef", clip_oid)
        a_mc = ET.SubElement(a_sc, "MasterClip")
        a_mc.set("ObjectURef", audio_master_uref)
        a_nm = ET.SubElement(a_sc, "Name")
        a_nm.text = clip_name
        a_ocg = ET.SubElement(a_sc, "OrigChGrp")
        a_ocg.text = "0"

        # AudioClipTrackItem
        acti = ET.SubElement(root, "AudioClipTrackItem")
        acti.set("ObjectID", item_oid)
        acti.set("ClassID", "d8273282-8805-11d5-af2d-9bfa89d4ddd4")
        acti.set("Version", "6")
        a_cti = ET.SubElement(acti, "ClipTrackItem")
        a_cti.set("Version", "8")
        a_co = ET.SubElement(a_cti, "ComponentOwner")
        a_co.set("Version", "1")
        a_comp = ET.SubElement(a_co, "Components")
        a_comp.set("ObjectRef", comp_oid)
        a_ti = ET.SubElement(a_cti, "TrackItem")
        a_ti.set("Version", "3")
        a_n = ET.SubElement(a_ti, "Node")
        a_n.set("Version", "1")
        a_nid = ET.SubElement(a_n, "ID")
        a_nid.text = str(int(item_oid) + 900000)
        a_s = ET.SubElement(a_ti, "Start")
        a_s.text = "0"
        a_e = ET.SubElement(a_ti, "End")
        a_e.text = str(seconds_to_ticks(100))
        a_sc_ref = ET.SubElement(a_cti, "SubClip")
        a_sc_ref.set("ObjectRef", subclip_oid)

        # AudioClipTrack
        act = ET.SubElement(root, "AudioClipTrack")
        act.set("ObjectUID", track_uid)
        act.set("ClassID", "f68dcd81-8805-11d5-af2d-9bfa89d4ddd5")
        act.set("Version", "1")
        a_ct = ET.SubElement(act, "ClipTrack")
        a_ct.set("Version", "2")
        a_trk = ET.SubElement(a_ct, "Track")
        a_trk.set("Version", "3")
        a_ci = ET.SubElement(a_ct, "ClipItems")
        a_ci.set("Version", "3")
        a_tis = ET.SubElement(a_ci, "TrackItems")
        a_tis.set("Version", "1")
        a_ti_ref = ET.SubElement(a_tis, "TrackItem")
        a_ti_ref.set("Index", "0")
        a_ti_ref.set("ObjectRef", item_oid)

    _make_audio_track("audio-track-uid-001", "70", "72", "75", "76", "host_mic.wav")
    _make_audio_track("audio-track-uid-002", "80", "82", "85", "86", "guest_mic.wav")

    # AudioTrackGroup
    atg = ET.SubElement(root, "AudioTrackGroup")
    atg.set("ObjectID", "11")
    atg.set("ClassID", "9e9abf7a-0918-49c2-91ae-991b5dde77cc")
    atg.set("Version", "13")
    atg_inner = ET.SubElement(atg, "TrackGroup")
    atg_inner.set("Version", "1")
    atg_tracks = ET.SubElement(atg_inner, "Tracks")
    atg_tracks.set("Version", "1")
    atr0 = ET.SubElement(atg_tracks, "Track")
    atr0.set("Index", "0")
    atr0.set("ObjectURef", "audio-track-uid-001")
    atr1 = ET.SubElement(atg_tracks, "Track")
    atr1.set("Index", "1")
    atr1.set("ObjectURef", "audio-track-uid-002")

    # Sequence
    seq = ET.SubElement(root, "Sequence")
    seq.set("ClassID", "6a15d903-8739-11d5-af2d-9b7855ad8974")
    tgs = ET.SubElement(seq, "TrackGroups")
    tgs.set("Version", "1")
    tg = ET.SubElement(tgs, "TrackGroup")
    tg.set("Version", "1")
    tg.set("Index", "0")
    first = ET.SubElement(tg, "First")
    first.text = "228cda18-3625-4d2d-951e-348879e4ed93"
    second = ET.SubElement(tg, "Second")
    second.set("ObjectRef", "10")
    # Audio TrackGroup ref
    tg_a = ET.SubElement(tgs, "TrackGroup")
    tg_a.set("Version", "1")
    tg_a.set("Index", "1")
    first_a = ET.SubElement(tg_a, "First")
    first_a.text = "audio-group-uuid"
    second_a = ET.SubElement(tg_a, "Second")
    second_a.set("ObjectRef", "11")
    nm = ET.SubElement(seq, "Name")
    nm.text = "TestSeq"

    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return gzip.compress(xml_str.encode("utf-8"))


def _rewrite_audio_track_indexes(prproj_bytes: bytes, indexes: list[int]) -> bytes:
    root = ET.fromstring(gzip.decompress(prproj_bytes).decode("utf-8"))
    atg = next(el for el in root if el.tag == "AudioTrackGroup")
    track_refs = atg.find("./TrackGroup/Tracks")
    refs = [el for el in track_refs if el.tag == "Track"]
    assert len(refs) == len(indexes)
    for ref, idx in zip(refs, indexes):
        ref.set("Index", str(idx))
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return gzip.compress(xml_str.encode("utf-8"))


class TestPatchSynthetic:
    def test_patch_creates_segments(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        cuts = [
            {"time": 30, "angle": 1},
            {"time": 70, "angle": 2},
        ]

        n = patch_prproj(in_path, cuts, "TestSeq", out_path)
        assert n == 3

        # Verify output
        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        obj_map = {}
        for el in root:
            oid = el.get("ObjectID")
            if oid:
                obj_map[oid] = el

        # Find track with the items
        items = []
        for el in root:
            if el.tag == "VideoClipTrack":
                track_items = el.find(".//TrackItems")
                if track_items is not None:
                    for ti in track_items:
                        ref = ti.get("ObjectRef")
                        if ref and ref in obj_map:
                            items.append(obj_map[ref])

        # Should have 3 track items (original + 2 new)
        multicam_items = []
        for item in items:
            if item.tag != "VideoClipTrackItem":
                continue
            sub_ref = item.find(".//SubClip").get("ObjectRef")
            sub = obj_map[sub_ref]
            clip_ref = sub.find("Clip").get("ObjectRef")
            clip = obj_map[clip_ref]
            is_mc = clip.find(".//IsMulticam")
            if is_mc is not None and is_mc.text == "true":
                start = int(item.find(".//Start").text)
                end = int(item.find(".//End").text)
                angle = int(clip.find(".//SelectedTrackIndex").text)
                multicam_items.append((start, end, angle))

        assert len(multicam_items) == 3
        # Check angles
        assert multicam_items[0][2] == 0  # original
        assert multicam_items[1][2] == 1
        assert multicam_items[2][2] == 2
        # Check continuity
        for i in range(len(multicam_items) - 1):
            assert multicam_items[i][1] == multicam_items[i + 1][0]

    def test_no_cuts_copies_file(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        n = patch_prproj(in_path, [], "TestSeq", out_path)
        assert n == 1
        assert out_path.exists()

    def test_wrong_sequence_name_raises(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        with pytest.raises(ValueError, match="Sequence.*not found"):
            patch_prproj(in_path, [{"time": 50, "angle": 1}], "Wrong", out_path)


class TestRoundTripGzip:
    def test_output_is_valid_gzip(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        patch_prproj(in_path, [{"time": 50, "angle": 1}], "TestSeq", out_path)

        with gzip.open(out_path, "rb") as f:
            data = f.read()
        root = ET.fromstring(data.decode("utf-8"))
        assert root.tag == "PremiereData"


REAL_FILE = Path("/Users/TeVaka/Downloads/A_NoCuts.prproj")


@pytest.mark.skipif(not REAL_FILE.exists(), reason="Real .prproj file not available")
class TestPatchRealFile:
    def test_patch_creates_correct_segments(self, tmp_path):
        out_path = tmp_path / "patched.prproj"
        cuts = [
            {"time": 502.92, "angle": 1},
            {"time": 972.48, "angle": 2},
            {"time": 1591.16, "angle": 0},
        ]

        n = patch_prproj(REAL_FILE, cuts, "No CUTS", out_path)
        assert n == 4

        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        obj_map = {}
        for el in root:
            oid = el.get("ObjectID")
            if oid:
                obj_map[oid] = el

        # Find the track and verify items
        for el in root:
            if el.tag == "VideoClipTrack":
                xml = ET.tostring(el, encoding="unicode")
                if "157" in xml:
                    track_items = el.find(".//TrackItems")
                    assert len(list(track_items)) == 4

                    angles = []
                    for ti_el in track_items:
                        ref = ti_el.get("ObjectRef")
                        item = obj_map[ref]
                        sub_ref = item.find(".//SubClip").get("ObjectRef")
                        sub = obj_map[sub_ref]
                        clip_ref = sub.find("Clip").get("ObjectRef")
                        clip = obj_map[clip_ref]
                        angles.append(int(clip.find(".//SelectedTrackIndex").text))

                    # Original angle was 2, then cuts: 1, 2, 0
                    assert angles == [2, 1, 2, 0]
                    break


class TestSegmentsToAudioIntervals:
    def test_empty(self):
        intervals = segments_to_audio_intervals([], 0)
        assert intervals == []

    def test_single_speaker_a_track0_active(self):
        segs = [_seg(0, 10, 0, SpeakerState.SPEAKER_A)]
        assert segments_to_audio_intervals(segs, 0) == [(0, 10)]

    def test_single_speaker_a_track1_gap(self):
        segs = [_seg(0, 10, 0, SpeakerState.SPEAKER_A)]
        assert segments_to_audio_intervals(segs, 1) == []

    def test_single_speaker_b_track1_active(self):
        segs = [_seg(0, 10, 1, SpeakerState.SPEAKER_B)]
        assert segments_to_audio_intervals(segs, 1) == [(0, 10)]

    def test_both_active_on_both_tracks(self):
        segs = [_seg(0, 10, 0, SpeakerState.BOTH)]
        assert segments_to_audio_intervals(segs, 0) == [(0, 10)]
        assert segments_to_audio_intervals(segs, 1) == [(0, 10)]

    def test_silence_keeps_both_tracks_active(self):
        """SILENCE should keep both tracks enabled to avoid audio holes."""
        segs = [_seg(0, 10, 0, SpeakerState.SILENCE)]
        assert segments_to_audio_intervals(segs, 0) == [(0, 10)]
        assert segments_to_audio_intervals(segs, 1) == [(0, 10)]

    def test_silence_between_speakers_no_gap(self):
        """SILENCE between speakers should not create audio holes."""
        segs = [
            _seg(0, 10, 0, SpeakerState.SPEAKER_A),
            _seg(10, 15, 0, SpeakerState.SILENCE),
            _seg(15, 25, 1, SpeakerState.SPEAKER_B),
        ]
        # Track 0: active during SPEAKER_A and SILENCE → merged [0,15]
        assert segments_to_audio_intervals(segs, 0) == [(0, 15)]
        # Track 1: active during SILENCE and SPEAKER_B → merged [10,25]
        assert segments_to_audio_intervals(segs, 1) == [(10, 25)]

    def test_alternating_speakers(self):
        segs = [
            _seg(0, 10, 0, SpeakerState.SPEAKER_A),
            _seg(10, 20, 1, SpeakerState.SPEAKER_B),
            _seg(20, 30, 0, SpeakerState.SPEAKER_A),
        ]
        # Track 0 (host): active at [0,10] and [20,30]
        assert segments_to_audio_intervals(segs, 0) == [(0, 10), (20, 30)]
        # Track 1 (guest): active at [10,20]
        assert segments_to_audio_intervals(segs, 1) == [(10, 20)]

    def test_adjacent_active_merged(self):
        segs = [
            _seg(0, 10, 0, SpeakerState.SPEAKER_A),
            _seg(10, 20, 0, SpeakerState.BOTH),  # host still active
        ]
        # Track 0: both segments active → merged
        assert segments_to_audio_intervals(segs, 0) == [(0, 20)]


def _parse_patched(out_path):
    """Parse a patched .prproj and return (root, obj_map)."""
    with gzip.open(out_path, "rb") as f:
        data = f.read().decode("utf-8")
    root = ET.fromstring(data)
    obj_map = {}
    for el in root:
        oid = el.get("ObjectID")
        if oid:
            obj_map[oid] = el
    return root, obj_map


def _get_audio_items_with_muted(root, obj_map):
    """Extract AudioClipTrackItem info with IsMuted state from parsed root.

    Returns dict: track_uid -> list of (start, end, is_muted) tuples.
    """
    items_by_track = {}
    for el in root:
        if el.tag == "AudioClipTrack":
            uid = el.get("ObjectUID")
            track_items = el.find(".//TrackItems")
            if track_items is None:
                continue
            items = []
            for ti_el in track_items:
                ref = ti_el.get("ObjectRef")
                if ref and ref in obj_map:
                    item = obj_map[ref]
                    start = int(item.find(".//Start").text)
                    end = int(item.find(".//End").text)
                    # Check IsMuted on ClipTrackItem
                    cti = item.find("ClipTrackItem")
                    muted_el = cti.find("IsMuted") if cti is not None else None
                    is_muted = muted_el is not None and muted_el.text == "true"
                    items.append((start, end, is_muted))
            items_by_track[uid] = items
    return items_by_track


class TestAudioMute:
    def _get_audio_items(self, root, obj_map):
        """Extract AudioClipTrackItem info from parsed root (legacy helper)."""
        full = _get_audio_items_with_muted(root, obj_map)
        return {uid: [(s, e) for s, e, _ in items] for uid, items in full.items()}

    def test_audio_mute_splits_tracks(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        # Segments: host speaks 0-30, guest speaks 30-70, both speak 70-100
        audio_segments = [
            _seg(0, 30, 0, SpeakerState.SPEAKER_A),
            _seg(30, 70, 1, SpeakerState.SPEAKER_B),
            _seg(70, 100, 0, SpeakerState.BOTH),
        ]

        cuts = [{"time": 30, "angle": 1}, {"time": 70, "angle": 0}]

        n = patch_prproj(
            in_path, cuts, "TestSeq", out_path,
            audio_segments=audio_segments,
        )
        assert n == 3

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        # Track 0 (host): active [0,30] and [70,100], disabled [30,70] → 3 items
        host_items = items_full.get("audio-track-uid-001", [])
        assert len(host_items) == 3
        # seg 0: enabled [0,30]
        assert host_items[0] == (seconds_to_ticks(0), seconds_to_ticks(30), False)
        # seg 1: disabled [30,70]
        assert host_items[1] == (seconds_to_ticks(30), seconds_to_ticks(70), True)
        # seg 2: enabled [70,100]
        assert host_items[2] == (seconds_to_ticks(70), seconds_to_ticks(100), False)

        # Track 1 (guest): active [30,100] (merged), disabled [0,30] → 2 items
        guest_items = items_full.get("audio-track-uid-002", [])
        assert len(guest_items) == 2
        # seg 0: disabled [0,30]
        assert guest_items[0] == (seconds_to_ticks(0), seconds_to_ticks(30), True)
        # seg 1: enabled [30,100]
        assert guest_items[1] == (seconds_to_ticks(30), seconds_to_ticks(100), False)

    def test_audio_split_shares_component_chain(self, tmp_path):
        """AudioComponentChain must be shared by reference across all segments
        of a track, not deep-copied per segment.

        Regression guard for the .prproj bloat that froze Premiere timelines
        on long projects: each audio segment used to clone the full chain, so
        the object count grew linearly with the number of segments.
        """
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        # The fixture has 2 audio tracks → 2 AudioComponentChain objects.
        root_in, _ = _parse_patched(in_path)
        n_chains_before = len(root_in.findall("AudioComponentChain"))
        assert n_chains_before == 2

        # Many alternating segments to exercise the per-segment split loop.
        audio_segments = [
            _seg(0, 15, 0, SpeakerState.SPEAKER_A),
            _seg(15, 30, 1, SpeakerState.SPEAKER_B),
            _seg(30, 45, 0, SpeakerState.SPEAKER_A),
            _seg(45, 60, 1, SpeakerState.SPEAKER_B),
            _seg(60, 75, 0, SpeakerState.SPEAKER_A),
            _seg(75, 100, 1, SpeakerState.SPEAKER_B),
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)
        chains_after = root.findall("AudioComponentChain")

        # The fix: chain count must NOT grow with the number of segments.
        assert len(chains_after) == n_chains_before, (
            f"AudioComponentChain count grew {n_chains_before} -> "
            f"{len(chains_after)}: chains are being duplicated per segment"
        )

        # Every split AudioClipTrackItem must reference one of the originals.
        chain_ids = {c.get("ObjectID") for c in chains_after}
        audio_items = [el for el in root if el.tag == "AudioClipTrackItem"]
        # The split really did produce more items than there are chains.
        assert len(audio_items) > n_chains_before
        for acti in audio_items:
            comp = acti.find(".//Components")
            assert comp is not None, "AudioClipTrackItem missing Components"
            assert comp.get("ObjectRef") in chain_ids, (
                "AudioClipTrackItem references a missing AudioComponentChain"
            )

    def test_audio_mute_silence_keeps_both_enabled(self, tmp_path):
        """All silence → both tracks stay enabled (no audio holes)."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        audio_segments = [
            _seg(0, 100, 0, SpeakerState.SILENCE),
        ]

        n = patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        # Both tracks: single enabled clip covering [0, 100]
        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        assert len(host_items) == 1
        assert host_items[0] == (seconds_to_ticks(0), seconds_to_ticks(100), False)
        assert len(guest_items) == 1
        assert guest_items[0] == (seconds_to_ticks(0), seconds_to_ticks(100), False)

    def test_explicit_audio_track_intervals_are_respected(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        n = patch_prproj(
            in_path,
            [],
            "TestSeq",
            out_path,
            audio_track_intervals_s={
                0: [(0.0, 30.0), (70.0, 100.0)],
                1: [(30.0, 100.0)],
            },
        )
        assert n == 1

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        assert host_items == [
            (seconds_to_ticks(0), seconds_to_ticks(30), False),
            (seconds_to_ticks(30), seconds_to_ticks(70), True),
            (seconds_to_ticks(70), seconds_to_ticks(100), False),
        ]
        assert guest_items == [
            (seconds_to_ticks(0), seconds_to_ticks(30), True),
            (seconds_to_ticks(30), seconds_to_ticks(100), False),
        ]

    def test_explicit_audio_track_intervals_support_pre_and_post_roll(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        n = patch_prproj(
            in_path,
            [],
            "TestSeq",
            out_path,
            audio_pre_roll_s=0.5,
            audio_post_roll_s=0.25,
            audio_track_intervals_s={
                0: [(10.0, 20.0)],
                1: [(30.0, 40.0)],
            },
        )
        assert n == 1

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        tpf_25 = fps_to_ticks_per_frame(25.0)
        assert host_items == [
            (0, snap_ticks_to_frame(seconds_to_ticks(9.5), tpf_25), True),
            (
                snap_ticks_to_frame(seconds_to_ticks(9.5), tpf_25),
                snap_ticks_to_frame(seconds_to_ticks(20.25), tpf_25),
                False,
            ),
            (snap_ticks_to_frame(seconds_to_ticks(20.25), tpf_25), seconds_to_ticks(100), True),
        ]
        assert guest_items == [
            (0, snap_ticks_to_frame(seconds_to_ticks(29.5), tpf_25), True),
            (
                snap_ticks_to_frame(seconds_to_ticks(29.5), tpf_25),
                snap_ticks_to_frame(seconds_to_ticks(40.25), tpf_25),
                False,
            ),
            (snap_ticks_to_frame(seconds_to_ticks(40.25), tpf_25), seconds_to_ticks(100), True),
        ]

    def test_explicit_audio_track_intervals_leave_unmapped_tracks_untouched(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        patch_prproj(
            in_path,
            [],
            "TestSeq",
            out_path,
            audio_track_intervals_s={
                0: [(0.0, 100.0)],
            },
        )

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        assert host_items == [
            (seconds_to_ticks(0), seconds_to_ticks(100), False),
        ]
        assert guest_items == [
            (seconds_to_ticks(0), seconds_to_ticks(100), False),
        ]

    def test_explicit_audio_track_intervals_use_real_sparse_track_indexes(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_rewrite_audio_track_indexes(_build_synthetic_prproj(), [0, 2]))

        patch_prproj(
            in_path,
            [],
            "TestSeq",
            out_path,
            audio_track_intervals_s={
                0: [(0.0, 30.0), (70.0, 100.0)],
                2: [(30.0, 100.0)],
            },
        )

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        assert host_items == [
            (seconds_to_ticks(0), seconds_to_ticks(30), False),
            (seconds_to_ticks(30), seconds_to_ticks(70), True),
            (seconds_to_ticks(70), seconds_to_ticks(100), False),
        ]
        assert guest_items == [
            (seconds_to_ticks(0), seconds_to_ticks(30), True),
            (seconds_to_ticks(30), seconds_to_ticks(100), False),
        ]

    def test_read_audio_offsets_preserves_sparse_track_indexes(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        in_path.write_bytes(_rewrite_audio_track_indexes(_build_synthetic_prproj(), [0, 2]))

        offsets = read_audio_offsets(in_path, "TestSeq")
        assert offsets == {
            0: 0.0,
            2: 0.0,
        }

    def test_no_audio_segments_leaves_audio_untouched(self, tmp_path):
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        n = patch_prproj(
            in_path, [{"time": 50, "angle": 1}], "TestSeq", out_path,
            audio_segments=None,
        )

        root, obj_map = _parse_patched(out_path)
        items = self._get_audio_items(root, obj_map)
        # Both tracks should still have exactly 1 item each (original)
        host_items = items.get("audio-track-uid-001", [])
        guest_items = items.get("audio-track-uid-002", [])
        assert len(host_items) == 1
        assert len(guest_items) == 1

    def test_audio_split_inpoint_invariants(self, tmp_path):
        """Verify InPoint/OutPoint are correct for non-zero orig_start."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        # Only host speaks 0-50
        audio_segments = [
            _seg(0, 50, 0, SpeakerState.SPEAKER_A),
            _seg(50, 100, 1, SpeakerState.SPEAKER_B),
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)

        # Check InPoint/OutPoint on all AudioClipTrackItem clips
        for el in root:
            if el.tag != "AudioClipTrackItem":
                continue
            start = int(el.find(".//Start").text)
            end = int(el.find(".//End").text)
            sub_ref = el.find(".//SubClip").get("ObjectRef")
            sub = obj_map[sub_ref]
            clip_ref = sub.find("Clip").get("ObjectRef")
            clip = obj_map[clip_ref]
            in_pt = int(clip.find(".//InPoint").text)
            out_pt = int(clip.find(".//OutPoint").text)

            # Duration must match
            assert (end - start) == (out_pt - in_pt), (
                f"Duration mismatch: tl={end-start}, src={out_pt-in_pt}"
            )
            # InPoint must be non-negative
            assert in_pt >= 0, f"Negative InPoint: {in_pt}"

    def test_audio_split_intervals_clipped_to_bounds(self, tmp_path):
        """Intervals extending past orig_end are clipped."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        # Create segments that extend past the clip's end (100s)
        audio_segments = [
            _seg(0, 50, 0, SpeakerState.SPEAKER_A),
            _seg(50, 150, 1, SpeakerState.SPEAKER_B),  # extends past 100s
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        # Track 0 (host): active [0,50], clip ends at 100 → disabled [50,100]
        host_items = items_full.get("audio-track-uid-001", [])
        assert len(host_items) == 2
        assert host_items[0][1] == seconds_to_ticks(50)
        assert host_items[1][1] == seconds_to_ticks(100)  # clipped, not 150

        # Track 1 (guest): active [50,100] (clipped from 150), disabled [0,50]
        guest_items = items_full.get("audio-track-uid-002", [])
        assert len(guest_items) == 2
        assert guest_items[1][1] == seconds_to_ticks(100)  # clipped

    def test_audio_disabled_flag_ismuted(self, tmp_path):
        """Verify IsMuted is set on disabled segments."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        audio_segments = [
            _seg(0, 50, 0, SpeakerState.SPEAKER_A),
            _seg(50, 100, 1, SpeakerState.SPEAKER_B),
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)

        # Check IsMuted on each AudioClipTrackItem's ClipTrackItem
        muted_states = []
        for el in root:
            if el.tag != "AudioClipTrackItem":
                continue
            cti = el.find("ClipTrackItem")
            muted_el = cti.find("IsMuted") if cti is not None else None
            is_muted = muted_el is not None and muted_el.text == "true"
            muted_states.append(is_muted)

        # Should have mix of muted and unmuted
        assert True in muted_states, "No muted segments found"
        assert False in muted_states, "No enabled segments found"

    def test_three_segment_split(self, tmp_path):
        """One long clip split into 3: seg0 enabled, seg1 disabled, seg2 enabled."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        audio_segments = [
            _seg(0, 30, 0, SpeakerState.SPEAKER_A),
            _seg(30, 70, 1, SpeakerState.SPEAKER_B),
            _seg(70, 100, 0, SpeakerState.SPEAKER_A),
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root, obj_map)

        # Track 0 (host): [0,30] enabled, [30,70] disabled, [70,100] enabled
        host_items = items_full.get("audio-track-uid-001", [])
        assert len(host_items) == 3
        assert host_items[0] == (seconds_to_ticks(0), seconds_to_ticks(30), False)
        assert host_items[1] == (seconds_to_ticks(30), seconds_to_ticks(70), True)
        assert host_items[2] == (seconds_to_ticks(70), seconds_to_ticks(100), False)

        # Verify InPoint/OutPoint for each segment
        # orig_in_point=0, orig_start=0 in synthetic prproj
        for el in root:
            if el.tag != "AudioClipTrackItem":
                continue
            start = int(el.find(".//Start").text)
            end = int(el.find(".//End").text)
            sub_ref = el.find(".//SubClip").get("ObjectRef")
            sub = obj_map[sub_ref]
            clip_ref = sub.find("Clip").get("ObjectRef")
            clip = obj_map[clip_ref]
            in_pt = int(clip.find(".//InPoint").text)
            out_pt = int(clip.find(".//OutPoint").text)
            # InPoint should equal Start (since orig_in_point=0, orig_start=0)
            assert in_pt == start
            assert out_pt == end

    def test_log_jsonl_output(self, tmp_path):
        """Verify JSONL log file is written with correct structure."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        log_path = tmp_path / "patched.prproj.log.jsonl"
        in_path.write_bytes(_build_synthetic_prproj())

        audio_segments = [
            _seg(0, 30, 0, SpeakerState.SPEAKER_A),
            _seg(30, 70, 1, SpeakerState.SPEAKER_B),
            _seg(70, 100, 0, SpeakerState.BOTH),
        ]

        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
            log_path=log_path,
        )

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines]

        # Should have speech_intervals, audio_split events, and summary
        event_types = [e["event"] for e in entries]
        assert "speech_intervals" in event_types
        assert "audio_split" in event_types
        assert "summary" in event_types

        # Check speech_intervals
        si = next(e for e in entries if e["event"] == "speech_intervals")
        assert si["count"] == 3
        assert len(si["intervals"]) == 3

        # Check summary has track counts
        summary = next(e for e in entries if e["event"] == "summary")
        assert "track_0_enabled" in summary
        assert "track_0_disabled" in summary
        assert "track_0_enabled_seconds" in summary
        assert summary["track_0_enabled"] > 0


class TestBuildFullSegments:
    def test_all_active(self):
        segs = _build_full_segments(0, 100, [(0, 100)])
        assert segs == [(0, 100, True)]

    def test_no_active(self):
        segs = _build_full_segments(0, 100, [])
        assert segs == [(0, 100, False)]

    def test_gap_before_and_after(self):
        segs = _build_full_segments(0, 100, [(20, 80)])
        assert segs == [(0, 20, False), (20, 80, True), (80, 100, False)]

    def test_two_active_with_gap(self):
        segs = _build_full_segments(0, 100, [(10, 30), (60, 90)])
        assert segs == [
            (0, 10, False),
            (10, 30, True),
            (30, 60, False),
            (60, 90, True),
            (90, 100, False),
        ]


class TestAudioDeepCopyRobustness:
    """Test that _split_audio_track_item works when Node/ID is missing."""

    def test_missing_node_id_no_crash(self, tmp_path):
        """Premiere versions without Node/ID in AudioClipTrackItem should not crash."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"

        # Build synthetic prproj and strip Node/ID from audio items
        data = _build_synthetic_prproj()
        xml_str = gzip.decompress(data).decode("utf-8")
        root = ET.fromstring(xml_str)

        # Remove Node elements from all AudioClipTrackItem elements
        for el in root:
            if el.tag == "AudioClipTrackItem":
                for ti in el.iter("TrackItem"):
                    node = ti.find("Node")
                    if node is not None:
                        ti.remove(node)

        # Re-save
        xml_out = ET.tostring(root, encoding="unicode", xml_declaration=True)
        in_path.write_bytes(gzip.compress(xml_out.encode("utf-8")))

        audio_segments = [
            _seg(0, 50, 0, SpeakerState.SPEAKER_A),
            _seg(50, 100, 1, SpeakerState.SPEAKER_B),
        ]

        # Should not raise
        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
        )

        root2, obj_map = _parse_patched(out_path)
        items_full = _get_audio_items_with_muted(root2, obj_map)

        # Both tracks should have 2 items each
        host_items = items_full.get("audio-track-uid-001", [])
        guest_items = items_full.get("audio-track-uid-002", [])
        assert len(host_items) == 2
        assert len(guest_items) == 2


class TestExpandAudioIntervals:
    def test_adjacent_merge_after_expansion(self):
        """Adjacent intervals should merge after expansion."""
        intervals = [(10.0, 20.0), (20.0, 30.0)]
        result = expand_audio_intervals(intervals, 0.1, 100.0)
        # After expansion: (9.9, 20.1) and (19.9, 30.1) → merged (9.9, 30.1)
        assert len(result) == 1
        assert abs(result[0][0] - 9.9) < 1e-6
        assert abs(result[0][1] - 30.1) < 1e-6

    def test_clamp_to_bounds(self):
        """Intervals should be clamped to [0, total_duration]."""
        intervals = [(0.0, 5.0), (95.0, 100.0)]
        result = expand_audio_intervals(intervals, 0.5, 100.0)
        assert result[0][0] == 0.0  # clamped, not -0.5
        assert result[-1][1] == 100.0  # clamped, not 100.5

    def test_zero_overlap_no_change(self):
        """overlap=0 returns intervals unchanged."""
        intervals = [(10.0, 20.0), (30.0, 40.0)]
        result = expand_audio_intervals(intervals, 0.0, 100.0)
        assert result == intervals

    def test_empty_intervals(self):
        """Empty input returns empty output."""
        assert expand_audio_intervals([], 0.1, 100.0) == []

    def test_gap_between_intervals_preserved(self):
        """Intervals with large gap stay separate after small expansion."""
        intervals = [(10.0, 20.0), (40.0, 50.0)]
        result = expand_audio_intervals(intervals, 0.1, 100.0)
        assert len(result) == 2
        assert abs(result[0][0] - 9.9) < 1e-6
        assert abs(result[0][1] - 20.1) < 1e-6
        assert abs(result[1][0] - 39.9) < 1e-6
        assert abs(result[1][1] - 50.1) < 1e-6

    def test_asymmetric_pre_roll_and_post_roll(self):
        """Pre-roll can be larger than post-roll to open speech earlier."""
        intervals = [(10.0, 20.0)]
        result = expand_audio_intervals(
            intervals,
            0.0,
            100.0,
            pre_roll_s=0.25,
            post_roll_s=0.10,
        )
        assert len(result) == 1
        assert abs(result[0][0] - 9.75) < 1e-6
        assert abs(result[0][1] - 20.1) < 1e-6


class TestFpsToTicksPerFrame:
    def test_23_976(self):
        tpf = fps_to_ticks_per_frame(23.976)
        # 254016000000 * 1001 // 24000 = 10594584000
        assert tpf == TICKS_PER_SECOND * 1001 // 24000

    def test_29_97(self):
        tpf = fps_to_ticks_per_frame(29.97)
        assert tpf == TICKS_PER_SECOND * 1001 // 30000

    def test_25(self):
        tpf = fps_to_ticks_per_frame(25)
        assert tpf == round(TICKS_PER_SECOND / 25)

    def test_30(self):
        tpf = fps_to_ticks_per_frame(30)
        assert tpf == round(TICKS_PER_SECOND / 30)


class TestSnapTicksToFrame:
    def test_exact_boundary_unchanged(self):
        tpf = fps_to_ticks_per_frame(23.976)
        val = tpf * 100
        assert snap_ticks_to_frame(val, tpf) == val

    def test_snaps_to_nearest(self):
        tpf = fps_to_ticks_per_frame(23.976)
        # Slightly past a boundary → snaps to that boundary
        assert snap_ticks_to_frame(tpf * 100 + 1, tpf) == tpf * 100
        # Just before next boundary → snaps to next
        assert snap_ticks_to_frame(tpf * 101 - 1, tpf) == tpf * 101

    def test_zero(self):
        tpf = fps_to_ticks_per_frame(29.97)
        assert snap_ticks_to_frame(0, tpf) == 0


class TestBuildSegmentsFrameSnapped:
    def test_cuts_aligned_to_frames(self):
        tpf = fps_to_ticks_per_frame(23.976)
        end = tpf * 1000  # exactly 1000 frames
        cuts = [
            {"time": 10.5, "angle": 1},
            {"time": 30.7, "angle": 2},
        ]
        segments = build_segments(0, end, 0, cuts, tpf=tpf)
        # All start/end values must be multiples of tpf
        for start, seg_end, _ in segments:
            assert start % tpf == 0, f"start {start} not aligned to tpf {tpf}"
            assert seg_end % tpf == 0, f"end {seg_end} not aligned to tpf {tpf}"

    def test_continuity_with_snapping(self):
        tpf = fps_to_ticks_per_frame(29.97)
        end = tpf * 500
        cuts = [
            {"time": 5.3, "angle": 1},
            {"time": 12.8, "angle": 0},
        ]
        segments = build_segments(0, end, 0, cuts, tpf=tpf)
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0]


class TestCloseCutsCollapse:
    def test_two_close_cuts_snap_to_same_frame(self):
        """Two cuts within one frame snap to the same tick → duplicate dropped."""
        tpf = fps_to_ticks_per_frame(23.976)
        frame_s = 1.0 / 23.976
        # Two cuts less than half a frame apart
        t = 10.0
        cuts = [
            {"time": t, "angle": 1},
            {"time": t + frame_s * 0.3, "angle": 2},
        ]
        end = tpf * 1000
        segments = build_segments(0, end, 0, cuts, tpf=tpf)
        # Second cut snaps to the same frame as first → produces no zero-length segment
        for start, seg_end, _ in segments:
            assert seg_end > start, f"zero-length segment: {start} == {seg_end}"

    def test_subframe_segment_merged(self):
        """Last segment < 1 frame (non-aligned orig_end) gets merged into previous."""
        tpf = fps_to_ticks_per_frame(23.976)
        # Place orig_end half a frame past a frame boundary so the last
        # segment is sub-frame when a cut snaps to that boundary.
        n_frames = 500
        orig_end = n_frames * tpf + tpf // 2  # half-frame overshoot
        # Cut that snaps to exactly frame N → last segment = tpf // 2 < tpf
        cut_time = (n_frames * tpf) / TICKS_PER_SECOND
        cuts = [{"time": cut_time, "angle": 1}]
        segments = build_segments(0, orig_end, 0, cuts, tpf=tpf)
        # Without merging we'd get 2 segments, last one = tpf//2.
        # With merging the sub-frame tail is absorbed → only 1 segment.
        assert len(segments) == 1
        assert segments[0] == (0, orig_end, 0)
        for start, seg_end, _ in segments:
            assert seg_end - start >= tpf, (
                f"sub-frame segment not merged: duration {seg_end - start} < tpf {tpf}"
            )


class TestReadSequenceTpf:
    def _build_minimal_sequence(self, frame_rate_text=None):
        """Build minimal XML with Sequence + VideoTrackGroup for tpf tests."""
        root = ET.Element("PremiereData")
        vtg = ET.SubElement(root, "VideoTrackGroup")
        vtg.set("ObjectID", "10")
        tg_inner = ET.SubElement(vtg, "TrackGroup")
        tg_inner.set("Version", "1")
        if frame_rate_text is not None:
            fr = ET.SubElement(tg_inner, "FrameRate")
            fr.text = frame_rate_text

        seq = ET.SubElement(root, "Sequence")
        tgs = ET.SubElement(seq, "TrackGroups")
        tg = ET.SubElement(tgs, "TrackGroup")
        second = ET.SubElement(tg, "Second")
        second.set("ObjectRef", "10")

        obj_map = {}
        for el in root:
            oid = el.get("ObjectID")
            if oid:
                obj_map[oid] = el
        return seq, obj_map

    def test_read_sequence_tpf(self):
        seq, obj_map = self._build_minimal_sequence("10160640000")
        assert _read_sequence_tpf(seq, obj_map) == 10160640000

    def test_read_sequence_tpf_missing(self):
        seq, obj_map = self._build_minimal_sequence(None)
        assert _read_sequence_tpf(seq, obj_map) == 0


class TestAutoDetectFps:
    def test_auto_detect_fps_snaps_correctly(self, tmp_path):
        """patch_prproj(fps=0.0) with FrameRate in prproj → all Start/End aligned."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        tpf_25 = 10160640000  # 25fps

        cuts = [
            {"time": 30, "angle": 1},
            {"time": 70, "angle": 2},
        ]
        n = patch_prproj(in_path, cuts, "TestSeq", out_path, fps=0.0)
        assert n == 3

        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        for el in root:
            if el.tag == "VideoClipTrackItem":
                start = int(el.find(".//Start").text)
                end = int(el.find(".//End").text)
                assert start % tpf_25 == 0, f"Start {start} not aligned to 25fps"
                assert end % tpf_25 == 0, f"End {end} not aligned to 25fps"

    def test_fps_override_takes_precedence(self, tmp_path):
        """patch_prproj(fps=30.0) → snaps to 30fps grid, ignoring FrameRate from XML."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        tpf_30 = fps_to_ticks_per_frame(30)

        cuts = [
            {"time": 30, "angle": 1},
            {"time": 70, "angle": 2},
        ]
        n = patch_prproj(in_path, cuts, "TestSeq", out_path, fps=30.0)
        assert n == 3

        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        for el in root:
            if el.tag == "VideoClipTrackItem":
                start = int(el.find(".//Start").text)
                end = int(el.find(".//End").text)
                assert start % tpf_30 == 0, f"Start {start} not aligned to 30fps"
                assert end % tpf_30 == 0, f"End {end} not aligned to 30fps"

    def test_inpoint_outpoint_frame_aligned(self, tmp_path):
        """InPoint/OutPoint of each video segment are aligned to tpf."""
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        tpf_25 = 10160640000

        cuts = [
            {"time": 30, "angle": 1},
            {"time": 70, "angle": 2},
        ]
        patch_prproj(in_path, cuts, "TestSeq", out_path, fps=0.0)

        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        obj_map = {}
        for el in root:
            oid = el.get("ObjectID")
            if oid:
                obj_map[oid] = el

        for el in root:
            if el.tag == "VideoClipTrackItem":
                subclip_ref = el.find(".//SubClip").get("ObjectRef")
                subclip = obj_map[subclip_ref]
                clip_ref = subclip.find("Clip").get("ObjectRef")
                clip = obj_map[clip_ref]
                clip_inner = clip.find("Clip") or clip
                in_point = int(clip_inner.find(".//InPoint").text)
                out_point = int(clip_inner.find(".//OutPoint").text)
                assert in_point % tpf_25 == 0, f"InPoint {in_point} not aligned"
                assert out_point % tpf_25 == 0, f"OutPoint {out_point} not aligned"


class TestSubframeAudioGap:
    def test_subframe_audio_gap_no_crash(self, tmp_path):
        """Sub-frame gaps between audio intervals don't crash with frame-snapping.

        When active intervals are frame-snapped but clip orig_start is not,
        a gap shorter than one frame can appear. InPoint/OutPoint snap would
        collapse it to zero duration, causing a duration mismatch RuntimeError.
        The merge logic should absorb sub-frame segments into neighbours.
        """
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        tpf_25 = 10160640000  # 25fps

        # Create audio segments where the first active interval starts at a
        # frame boundary slightly after 0, leaving a sub-frame gap.
        # The gap (0.001s ≈ 254,016,000 ticks) is much less than 1 frame
        # (10,160,640,000 ticks at 25fps).
        gap_s = 0.001  # 1ms — sub-frame at 25fps
        audio_segments = [
            _seg(gap_s, 50, 0, SpeakerState.SPEAKER_A),
            _seg(50, 100, 1, SpeakerState.SPEAKER_B),
        ]

        # Without the sub-frame merge fix, this raises:
        # RuntimeError: Audio segment N: duration mismatch src=0 tl=...
        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
            fps=0.0,
        )

        # Verify output is valid
        with gzip.open(out_path, "rb") as f:
            data = f.read().decode("utf-8")
        root = ET.fromstring(data)

        # All audio segments should have positive duration
        for el in root:
            if el.tag == "AudioClipTrackItem":
                start = int(el.find(".//Start").text)
                end = int(el.find(".//End").text)
                assert end > start, f"Zero/negative duration audio segment: {start}..{end}"

    def test_independent_snap_duration_mismatch(self, tmp_path):
        """Snapping InPoint and OutPoint independently must not cause duration mismatch.

        When orig_start is not frame-aligned, the computed InPoint and OutPoint
        snap to different frame boundaries, making src_duration != tl_duration.
        The fix: only snap InPoint, derive OutPoint = InPoint + tl_duration.
        """
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(_build_synthetic_prproj())

        # Two long segments — no sub-frame merge possible, but snap still
        # causes mismatch if InPoint and OutPoint are snapped independently.
        # Use segments far enough apart that the sub-frame merge won't kick in.
        audio_segments = [
            _seg(0, 30, 0, SpeakerState.SPEAKER_A),
            _seg(30, 60, 1, SpeakerState.SPEAKER_B),
        ]

        # Without the fix, this raises:
        # RuntimeError: Audio segment N: duration mismatch src=... tl=...
        patch_prproj(
            in_path, [], "TestSeq", out_path,
            audio_segments=audio_segments,
            fps=0.0,
        )

    def test_video_independent_snap_duration_mismatch(self, tmp_path):
        """Video InPoint/OutPoint must not be snapped independently.

        When orig_in_point is not frame-aligned, independent snapping of
        InPoint and OutPoint can produce src_duration != tl_duration (±1 frame).
        The fix: snap only InPoint, derive OutPoint = InPoint + tl_duration.
        """
        # Build prproj with non-frame-aligned InPoint
        data = _build_synthetic_prproj()
        root = ET.fromstring(gzip.decompress(data).decode("utf-8"))

        # Shift InPoint by half a frame (non-aligned offset)
        tpf = 10160640000  # 25fps
        half_frame = tpf // 2
        for vc in root.iter("VideoClip"):
            inp_el = vc.find(".//InPoint")
            outp_el = vc.find(".//OutPoint")
            if inp_el is not None:
                inp_el.text = str(half_frame)
            if outp_el is not None:
                outp_el.text = str(half_frame + seconds_to_ticks(100))

        modified = gzip.compress(
            ET.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8")
        )
        in_path = tmp_path / "test.prproj"
        out_path = tmp_path / "patched.prproj"
        in_path.write_bytes(modified)

        cuts = [
            {"time": 30.0, "angle": 1},
            {"time": 60.0, "angle": 0},
        ]

        patch_prproj(in_path, cuts, "TestSeq", out_path, fps=0.0)

        # Verify: for every video segment, src_duration == tl_duration
        with gzip.open(out_path, "rb") as f:
            patched_root = ET.fromstring(f.read().decode("utf-8"))

        for el in patched_root:
            if el.tag == "VideoClipTrackItem":
                start = int(el.find(".//Start").text)
                end = int(el.find(".//End").text)
                tl_duration = end - start
                # Find the associated clip's InPoint/OutPoint
                cti = el.find("ClipTrackItem")
                sc = cti.find("SubClip")
                clip_ref = sc.get("ObjectRef")
                # Find SubClip -> Clip -> VideoClip
                subclip = None
                for sub in patched_root:
                    if sub.tag == "SubClip" and sub.get("ObjectID") == clip_ref:
                        subclip = sub
                        break
                assert subclip is not None
                vc_ref = subclip.find("Clip").get("ObjectRef")
                video_clip = None
                for vc in patched_root:
                    if vc.tag == "VideoClip" and vc.get("ObjectID") == vc_ref:
                        video_clip = vc
                        break
                assert video_clip is not None
                in_pt = int(video_clip.find(".//InPoint").text)
                out_pt = int(video_clip.find(".//OutPoint").text)
                src_duration = out_pt - in_pt
                assert src_duration == tl_duration, (
                    f"Video segment duration mismatch: "
                    f"src={src_duration} tl={tl_duration} "
                    f"(InPoint={in_pt}, OutPoint={out_pt}, "
                    f"Start={start}, End={end})"
                )
