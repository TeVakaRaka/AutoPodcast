"""Speaker role normalization."""

from __future__ import annotations

_ROLE_MAP = {
    "host": "host",
    "presenter": "host",
    "anchor": "host",
    "ведущий": "host",
    "ведущая": "host",
    "guest": "guest",
    "interviewee": "guest",
    "participant": "guest",
    "гость": "guest",
    "гостья": "guest",
}


def normalize_speaker_role(label: str) -> str:
    """Normalize a speaker role label to a canonical form.

    Supports English and Russian aliases. Unknown labels pass through unchanged.
    """
    return _ROLE_MAP.get(label.strip().lower(), label)
