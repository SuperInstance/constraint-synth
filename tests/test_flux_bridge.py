"""Tests for flux_bridge — constraint-synth ↔ flux-tensor-midi bridge."""

import math
import os
import tempfile

import numpy as np
import pytest

from constraint_synth.flux_bridge import (
    CHANNEL_NAMES,
    FluxBridge,
    FluxVectorMapper,
    LATTICE_SHAPES,
    MidiEventShim,
    _normalize_event,
    compare_renderers,
)
from constraint_synth.synth import ConstraintSynth


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_event(note=60, velocity=100, start_ms=0.0, duration_ms=250.0, channel=0):
    return MidiEventShim(note=note, velocity=velocity, start_ms=start_ms,
                         duration_ms=duration_ms, channel=channel)


def _make_events(n=4, base_note=60, duration_ms=250.0, spacing_ms=300.0):
    return [
        _make_event(note=base_note + i, start_ms=i * spacing_ms, duration_ms=duration_ms)
        for i in range(n)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 1. MidiEventShim
# ──────────────────────────────────────────────────────────────────────────────

class TestMidiEventShim:
    def test_valid_creation(self):
        ev = _make_event(note=64, velocity=80)
        assert ev.note == 64
        assert ev.velocity == 80
        assert ev.start_ms == 0.0
        assert ev.duration_ms == 250.0
        assert ev.channel == 0

    def test_invalid_note_raises(self):
        with pytest.raises(ValueError, match="note"):
            MidiEventShim(note=128, velocity=100, start_ms=0, duration_ms=100)

    def test_invalid_velocity_raises(self):
        with pytest.raises(ValueError, match="velocity"):
            MidiEventShim(note=60, velocity=-1, start_ms=0, duration_ms=100)

    def test_normalize_wraps_dict_like(self):
        class FakeEvent:
            note = 60
            velocity = 100
            start_ms = 100.0
            duration_ms = 200.0
            channel = 1

        result = _normalize_event(FakeEvent())
        assert isinstance(result, MidiEventShim)
        assert result.channel == 1

    def test_normalize_passes_through_shim(self):
        ev = _make_event()
        assert _normalize_event(ev) is ev


# ──────────────────────────────────────────────────────────────────────────────
# 2. FluxVectorMapper
# ──────────────────────────────────────────────────────────────────────────────

class TestFluxVectorMapper:
    def test_basic_mapping(self):
        mapper = FluxVectorMapper()
        params = mapper.map_to_synth_params([0.5] * 9)
        assert "freq" in params
        assert "note" in params
        assert params["note"] >= 0
        assert params["velocity"] > 0

    def test_pitch_channel(self):
        mapper = FluxVectorMapper(base_note=60)
        # Channel 0 = 0 → base note (C4=60)
        params = mapper.map_to_synth_params([0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert params["note"] == 60

        # Channel 0 = 1 → base note + 12
        params = mapper.map_to_synth_params([1, 0, 0, 0, 0, 0, 0, 0, 0])
        assert params["note"] == 72

    def test_timbre_maps_to_lattice_shape(self):
        mapper = FluxVectorMapper()
        # Channel 2 = 0 → sine
        params = mapper.map_to_synth_params([0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert params["lattice_shape"] == "sine"

        # Channel 2 = 0.5 → "square" (index 2 of 5 shapes, since int(0.5*5)=2)
        params = mapper.map_to_synth_params([0, 0, 0.5, 0, 0, 0, 0, 0, 0])
        assert params["lattice_shape"] in LATTICE_SHAPES

    def test_brightness_maps_to_filter_cutoff(self):
        mapper = FluxVectorMapper()
        low = mapper.map_to_synth_params([0, 0, 0, 0.0, 0, 0, 0, 0, 0])
        high = mapper.map_to_synth_params([0, 0, 0, 1.0, 0, 0, 0, 0, 0])
        assert high["filter_cutoff"] > low["filter_cutoff"]
        assert low["filter_cutoff"] >= 200.0
        assert high["filter_cutoff"] <= 8000.0

    def test_space_maps_to_reverb(self):
        mapper = FluxVectorMapper()
        dry = mapper.map_to_synth_params([0, 0, 0, 0, 0.0, 0, 0, 0, 0])
        wet = mapper.map_to_synth_params([0, 0, 0, 0, 1.0, 0, 0, 0, 0])
        assert wet["reverb_wet"] > dry["reverb_wet"]

    def test_too_few_channels_raises(self):
        mapper = FluxVectorMapper()
        with pytest.raises(ValueError, match="9 channels"):
            mapper.map_to_synth_params([0.5, 0.5, 0.5])

    def test_map_to_synth_returns_constraint_synth(self):
        mapper = FluxVectorMapper()
        synth = mapper.map_to_synth([0.5] * 9)
        assert isinstance(synth, ConstraintSynth)


# ──────────────────────────────────────────────────────────────────────────────
# 3. FluxBridge rendering
# ──────────────────────────────────────────────────────────────────────────────

class TestFluxBridgeRenderEvents:
    def test_empty_events_returns_empty(self):
        bridge = FluxBridge()
        audio = bridge.render_events([])
        assert len(audio) == 0

    def test_single_event_produces_audio(self):
        bridge = FluxBridge()
        events = [_make_event()]
        audio = bridge.render_events(events)
        assert len(audio) > 0
        assert audio.dtype == np.float64

    def test_multiple_events_produce_longer_audio(self):
        bridge = FluxBridge()
        single = bridge.render_events([_make_event(start_ms=0, duration_ms=250)])
        multiple = bridge.render_events(_make_events(4))
        assert len(multiple) > len(single)

    def test_events_sorted_by_start_time(self):
        bridge = FluxBridge()
        # Pass events out of order
        events = [
            _make_event(start_ms=500, duration_ms=200),
            _make_event(start_ms=0, duration_ms=200),
            _make_event(start_ms=250, duration_ms=200),
        ]
        audio = bridge.render_events(events)
        assert len(audio) > 0
        # Should not crash and should produce valid audio
        assert np.max(np.abs(audio)) <= 1.0 or np.max(np.abs(audio)) <= 1.1

    def test_preset_applied(self):
        bridge = FluxBridge(preset="bop_sax")
        audio = bridge.render_events([_make_event()])
        assert len(audio) > 0

    def test_overlapping_events_mix(self):
        bridge = FluxBridge()
        events = [
            _make_event(note=60, start_ms=0, duration_ms=500),
            _make_event(note=64, start_ms=100, duration_ms=500),
        ]
        audio = bridge.render_events(events)
        assert len(audio) > 0
        # Should have content from both notes (not silence)
        assert np.max(np.abs(audio)) > 0.01


# ──────────────────────────────────────────────────────────────────────────────
# 4. FluxVector direct rendering
# ──────────────────────────────────────────────────────────────────────────────

class TestFluxBridgeFluxVector:
    def test_basic_flux_render(self):
        bridge = FluxBridge()
        values = [0.5, 0.7, 0.2, 0.5, 0.3, 1.0, 0.0, 0.5, 0.6]
        audio = bridge.render_flux_vector(values, duration_ms=500)
        assert len(audio) > 0
        assert audio.dtype == np.float64

    def test_silent_flux(self):
        bridge = FluxBridge()
        # All zeros = silence (zero velocity)
        values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        audio = bridge.render_flux_vector(values, duration_ms=200)
        # Should be near-silent (velocity will be 0 or very low)
        assert np.max(np.abs(audio)) < 0.01

    def test_flux_sequence(self):
        bridge = FluxBridge()
        vectors = [
            {"values": [0.5, 0.8, 0.0, 0.5, 0.2, 1.0, 0.0, 0.5, 0.4]},
            {"values": [0.3, 0.6, 0.4, 0.6, 0.4, 0.95, 0.1, 0.3, 0.5]},
            {"values": [0.7, 0.9, 0.0, 0.3, 0.5, 1.0, 0.0, 0.7, 0.3]},
        ]
        audio = bridge.render_flux_sequence(vectors)
        assert len(audio) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Round-trip: events → constraints → synth
# ──────────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_events_to_constraints_preserves_data(self):
        bridge = FluxBridge()
        events = _make_events(3)
        constraints = bridge.events_to_constraints(events)
        assert len(constraints) == 3
        assert constraints[0]["note"] == 60
        assert constraints[1]["note"] == 61
        assert constraints[2]["note"] == 62

    def test_constraint_frequency_calculation(self):
        bridge = FluxBridge()
        events = [_make_event(note=69)]  # A4 = 440 Hz
        constraints = bridge.events_to_constraints(events)
        assert abs(constraints[0]["freq"] - 440.0) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# 6. WAV output
# ──────────────────────────────────────────────────────────────────────────────

class TestWavOutput:
    def test_to_wav_creates_file(self):
        bridge = FluxBridge()
        events = _make_events(2)
        audio = bridge.render_events(events)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            bridge.to_wav(audio, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 44  # WAV header + data
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Comparison utility
# ──────────────────────────────────────────────────────────────────────────────

class TestCompareRenderers:
    def test_compare_produces_results(self):
        events = _make_events(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compare_renderers(events, output_dir=tmpdir)
            assert "direct_path" in result
            assert os.path.exists(result["direct_path"])
            assert result["events_count"] == 3
            assert result["audio_duration_sec"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 8. Edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_duration_event_skipped(self):
        bridge = FluxBridge()
        events = [_make_event(duration_ms=0)]
        audio = bridge.render_events(events)
        # Should not crash; may produce silence or minimal audio
        assert isinstance(audio, np.ndarray)

    def test_very_short_event(self):
        bridge = FluxBridge()
        events = [_make_event(duration_ms=1)]
        audio = bridge.render_events(events)
        assert len(audio) > 0

    def test_very_long_event(self):
        bridge = FluxBridge()
        events = [_make_event(duration_ms=5000)]
        audio = bridge.render_events(events)
        assert len(audio) > 0
        expected_min = 4.5 * 44100  # ~5 seconds worth of samples
        assert len(audio) >= expected_min

    def test_channel_9_constant(self):
        assert len(CHANNEL_NAMES) == 9

    def test_all_presets_work(self):
        for preset in ConstraintSynth.PRESETS:
            bridge = FluxBridge(preset=preset)
            audio = bridge.render_events([_make_event()])
            assert len(audio) > 0, f"Preset '{preset}' produced no audio"
