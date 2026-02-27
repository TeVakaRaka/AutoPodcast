"""JSON export/import for Timeline data."""

from __future__ import annotations

import json
from pathlib import Path

from autopodcast.models.domain import (
    DuckingEvent,
    Segment,
    SpeakerState,
    Timeline,
)


def timeline_to_dict(timeline: Timeline) -> dict:
    """Convert Timeline to a JSON-serializable dict."""
    return {
        "total_duration_s": timeline.total_duration_s,
        "fps": timeline.fps,
        "segments": [
            {
                "start_s": s.start_s,
                "end_s": s.end_s,
                "camera_index": s.camera_index,
                "speaker_state": s.speaker_state.value,
                "speaker_label": s.speaker_label,
            }
            for s in timeline.segments
        ],
        "ducking_events": [
            {
                "time_s": e.time_s,
                "target_db": e.target_db,
                "track_index": e.track_index,
            }
            for e in timeline.ducking_events
        ],
    }


def dict_to_timeline(data: dict) -> Timeline:
    """Reconstruct Timeline from a dict (loaded from JSON)."""
    segments = [
        Segment(
            start_s=s["start_s"],
            end_s=s["end_s"],
            camera_index=s["camera_index"],
            speaker_state=SpeakerState(s["speaker_state"]),
            speaker_label=s.get("speaker_label"),
        )
        for s in data["segments"]
    ]

    ducking_events = [
        DuckingEvent(
            time_s=e["time_s"],
            target_db=e["target_db"],
            track_index=e["track_index"],
        )
        for e in data.get("ducking_events", [])
    ]

    return Timeline(
        segments=segments,
        ducking_events=ducking_events,
        total_duration_s=data["total_duration_s"],
        fps=data.get("fps", 29.97),
    )


def save_timeline(timeline: Timeline, path: Path) -> None:
    """Save timeline to a JSON file."""
    data = timeline_to_dict(timeline)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_timeline(path: Path) -> Timeline:
    """Load timeline from a JSON file."""
    data = json.loads(path.read_text())
    return dict_to_timeline(data)
