"""Tests for the GUI form specification and CLI argument builder.

No tkinter import here — gui.spec is pure data + logic.
"""

from __future__ import annotations

import pytest

from autopodcast.cli import cli
from autopodcast.gui.spec import (
    MODES,
    build_argv,
    mode_by_key,
)


def _filled(spec, **overrides):
    """Build a values dict with every field at its default (required fields
    get a plausible path/name), then apply overrides."""
    vals = {}
    for f in spec.fields:
        if f.kind == "bool":
            vals[f.arg] = f.default
        elif f.required:
            if f.arg == "--in":
                vals[f.arg] = "/proj/Episode.prproj"
            elif f.kind == "file":
                vals[f.arg] = "/proj/mic.wav"
            else:
                vals[f.arg] = "Sequence 01"
        else:
            vals[f.arg] = "" if f.default is None else f.default
    vals.update(overrides)
    return vals


# ---------------------------------------------------------------------------
# build_argv — defaults
# ---------------------------------------------------------------------------


class TestBuildArgvDefaults:
    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_defaults_emit_only_required_plus_out(self, spec):
        """With everything at default, argv = cmd + required flags + --out."""
        argv = build_argv(spec, _filled(spec))
        assert argv[0] == spec.cmd
        # --out is always derived
        assert "--out" in argv
        out_idx = argv.index("--out")
        assert argv[out_idx + 1].endswith(spec.out_suffix + ".prproj")
        # no optional/advanced flag leaked through at default values
        for f in spec.fields:
            if not f.required and f.kind != "bool":
                assert f.arg not in argv, f"{f.arg} leaked at default"

    def test_multicam_default_argv_exact(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec))
        assert argv == [
            "auto-multicam",
            "--in", "/proj/Episode.prproj",
            "--mic-a", "/proj/mic.wav",
            "--mic-b", "/proj/mic.wav",
            "--seq", "Sequence 01",
            "--out", "/proj/Episode_multicam.prproj",
        ]


# ---------------------------------------------------------------------------
# build_argv — overrides
# ---------------------------------------------------------------------------


class TestBuildArgvOverrides:
    def test_changed_int_is_emitted(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec, **{"--camera-host": 5}))
        assert "--camera-host" in argv
        assert argv[argv.index("--camera-host") + 1] == "5"

    def test_changed_float_is_emitted(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec, **{"--speech-threshold": -20.0}))
        assert "--speech-threshold" in argv
        assert argv[argv.index("--speech-threshold") + 1] == "-20.0"

    def test_bool_toggled_off_emits_off_flag(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec, **{"--mute-audio": False}))
        assert "--no-mute-audio" in argv
        assert "--mute-audio" not in argv

    def test_bool_at_default_emits_nothing(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec))  # mute-audio default True
        assert "--mute-audio" not in argv
        assert "--no-mute-audio" not in argv

    def test_cross_cancel_off_emits_no_cross_cancel(self):
        spec = mode_by_key("multicam")
        argv = build_argv(spec, _filled(spec, **{"--cross-cancel": False}))
        assert "--no-cross-cancel" in argv

    def test_motion_check_on_emits_motion_check(self):
        # 4cams motion-check default is False -> turning it on emits the flag
        spec = mode_by_key("4cams")
        argv = build_argv(spec, _filled(spec, **{"--motion-check": True}))
        assert "--motion-check" in argv

    def test_optional_choice_changed_is_emitted(self):
        spec = mode_by_key("4cams")
        argv = build_argv(spec, _filled(spec, **{"--motion-hwaccel": "cpu"}))
        assert "--motion-hwaccel" in argv
        assert argv[argv.index("--motion-hwaccel") + 1] == "cpu"


# ---------------------------------------------------------------------------
# build_argv — output derivation and validation
# ---------------------------------------------------------------------------


class TestBuildArgvOutput:
    def test_out_keeps_directory_and_stem(self):
        spec = mode_by_key("monologue")
        argv = build_argv(spec, _filled(spec, **{"--in": "/a/b/My Talk.prproj"}))
        assert argv[argv.index("--out") + 1] == "/a/b/My Talk_monologue.prproj"

    def test_missing_required_raises(self):
        spec = mode_by_key("multicam")
        vals = _filled(spec)
        vals["--seq"] = ""
        with pytest.raises(ValueError, match="Имя секвенции"):
            build_argv(spec, vals)

    def test_sakha_optional_mics_omitted_when_empty(self):
        """SAKHA mics are optional (auto-resolved) — empty -> not in argv."""
        spec = mode_by_key("sakha")
        argv = build_argv(spec, _filled(spec))
        for arg in ("--mic-main-host", "--mic-cohost", "--mic-guest", "--xml"):
            assert arg not in argv

    def test_sakha_optional_mic_emitted_when_set(self):
        spec = mode_by_key("sakha")
        argv = build_argv(spec, _filled(spec, **{"--mic-guest": "/proj/g.wav"}))
        assert "--mic-guest" in argv
        assert argv[argv.index("--mic-guest") + 1] == "/proj/g.wav"


# ---------------------------------------------------------------------------
# Slider fields
# ---------------------------------------------------------------------------


class TestSliderFields:
    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_slider_ranges_are_valid(self, spec):
        for f in spec.fields:
            if not f.is_slider:
                continue
            assert f.vmin < f.vmax, f"{spec.cmd}:{f.arg} vmin must be < vmax"
            assert f.vstep > 0, f"{spec.cmd}:{f.arg} vstep must be > 0"
            if f.default is not None:
                assert f.vmin <= f.default <= f.vmax, (
                    f"{spec.cmd}:{f.arg} default {f.default} outside slider range"
                )
            steps = (f.vmax - f.vmin) / f.vstep
            assert abs(steps - round(steps)) < 1e-6, (
                f"{spec.cmd}:{f.arg} range is not a whole multiple of vstep"
            )

    def test_at_least_one_slider_per_creative_mode(self):
        # sakha, monologue and 4cams expose tempo/percentage sliders
        for key in ("sakha", "monologue", "4cams", "multicam"):
            spec = mode_by_key(key)
            assert any(f.is_slider for f in spec.fields), f"{key} has no sliders"


# ---------------------------------------------------------------------------
# Consistency: spec defaults must match the CLI @click.option defaults
# ---------------------------------------------------------------------------


def _find_param(command, flag):
    for p in command.params:
        if flag in getattr(p, "opts", []):
            return p
    return None


class TestSpecMatchesCli:
    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_command_exists(self, spec):
        assert spec.cmd in cli.commands

    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_every_field_arg_is_a_real_cli_option(self, spec):
        command = cli.commands[spec.cmd]
        for f in spec.fields:
            assert _find_param(command, f.arg) is not None, (
                f"{spec.cmd}: field {f.arg} is not a CLI option"
            )

    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_field_defaults_match_cli_defaults(self, spec):
        command = cli.commands[spec.cmd]
        for f in spec.fields:
            param = _find_param(command, f.arg)
            if f.required:
                continue  # required options have no meaningful default
            cli_default = param.default
            # click represents "no default" either as None or as an
            # UNSET sentinel (click >= 8.2); treat both as "no default".
            if f.default is None:
                assert cli_default is None or "UNSET" in repr(cli_default), (
                    f"{spec.cmd}: {f.arg} expected no CLI default, got {cli_default!r}"
                )
            else:
                assert cli_default == f.default, (
                    f"{spec.cmd}: {f.arg} default mismatch — "
                    f"spec={f.default!r} cli={cli_default!r}"
                )

    @pytest.mark.parametrize("spec", MODES, ids=[m.key for m in MODES])
    def test_bool_flag_pairs_match_cli(self, spec):
        command = cli.commands[spec.cmd]
        for f in spec.fields:
            if f.kind != "bool":
                continue
            param = _find_param(command, f.arg)
            assert f.flag_pair[0] in param.opts
            assert f.flag_pair[1] in param.secondary_opts
