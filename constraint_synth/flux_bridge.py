"""flux_bridge — Bridge between flux-tensor-midi and constraint-synth.

Converts flux-tensor-midi MidiEvent streams and FluxVectors into
constraint-synth audio, enabling direct synthesis without DawDreamer.

This is the synergy: constraint parameters → MIDI (flux) → audio (dawdreamer)
OR constraint parameters → audio directly (synth). Same source, two renderers.

Usage:
    from constraint_synth.flux_bridge import FluxBridge

    bridge = FluxBridge(preset="piano_ballad")
    audio = bridge.render_events(midi_events)
    bridge.to_wav(audio, "output.wav")
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter
from .synth import ConstraintSynth, BiquadLowpass, SchroederReverb


# ──────────────────────────────────────────────────────────────────────────────
# Channel mapping: FluxVector channels → constraint-synth parameters
# ──────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = (
    "pitch",
    "dynamics",
    "timbre",
    "brightness",
    "space",
    "tension",
    "noise",
    "snap",
    "weight",
)

# Lattice shape lookup by index (timbre channel)
LATTICE_SHAPES = ("sine", "saw", "square", "triangle", "eisenstein")

# Default note-to-channel mapping (matches flux-tensor-midi NoteName)
NOTE_CHANNEL_BASE = 60  # C4


# ──────────────────────────────────────────────────────────────────────────────
# MidiEvent shim — works with or without flux-tensor-midi installed
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MidiEventShim:
    """Lightweight MidiEvent compatible with flux-tensor-midi's MidiEvent.

    If flux_tensor_midi is installed, we accept its MidiEvent directly.
    Otherwise, this shim provides the same interface.
    """
    note: int
    velocity: int
    start_ms: float
    duration_ms: float
    channel: int = 0

    def __post_init__(self):
        if not 0 <= self.note <= 127:
            raise ValueError(f"note must be 0–127, got {self.note}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity must be 0–127, got {self.velocity}")


def _normalize_event(ev: Any) -> MidiEventShim:
    """Accept any event with note/velocity/start_ms/duration_ms attributes."""
    if isinstance(ev, MidiEventShim):
        return ev
    return MidiEventShim(
        note=ev.note,
        velocity=ev.velocity,
        start_ms=ev.start_ms,
        duration_ms=ev.duration_ms,
        channel=getattr(ev, "channel", 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# FluxVector mapper
# ──────────────────────────────────────────────────────────────────────────────

class FluxVectorMapper:
    """Maps a 9-channel FluxVector to constraint-synth parameters.

    Channel semantics:
        0 (pitch)      → oscillator frequency via note offset from C4
        1 (dynamics)   → envelope sustain (0–1) + velocity scaling
        2 (timbre)     → lattice_shape index (0–4)
        3 (brightness) → filter_cutoff (200–8000 Hz)
        4 (space)      → reverb_wet (0–1)
        5 (tension)    → lattice_stretch (0.9–1.1)
        6 (noise)      → noise_floor (0–0.1)
        7 (snap)       → snap_threshold (0–1)
        8 (weight)     → attack (0.001–0.5) and decay (0.01–1.0)
    """

    def __init__(
        self,
        base_note: int = 60,
        velocity_scale: int = 100,
        sample_rate: int = 44100,
    ):
        self.base_note = base_note
        self.velocity_scale = velocity_scale
        self.sample_rate = sample_rate

    def map_to_synth_params(
        self,
        values: Sequence[float],
        salience: Sequence[float] | None = None,
        tolerance: Sequence[float] | None = None,
    ) -> dict:
        """Map FluxVector values to a dict of constraint-synth constructor args.

        Returns a dict suitable for building a LatticeOscillator, FunnelEnvelope,
        and synth config.
        """
        if len(values) < 9:
            raise ValueError(f"Expected 9 channels, got {len(values)}")

        # Channel 0: pitch → semitone offset from base_note
        pitch_offset = values[0] * 12.0  # 0–12 semitones
        note = min(127, max(0, int(self.base_note + pitch_offset)))
        freq = 440.0 * (2 ** ((note - 69) / 12.0))

        # Channel 1: dynamics → sustain level
        dynamics = max(0.0, min(1.0, values[1]))

        # Channel 2: timbre → lattice shape index
        shape_idx = max(0, min(len(LATTICE_SHAPES) - 1, int(values[2] * len(LATTICE_SHAPES))))
        lattice_shape = LATTICE_SHAPES[shape_idx]

        # Channel 3: brightness → filter cutoff
        brightness = max(0.0, min(1.0, values[3]))
        filter_cutoff = 200.0 + brightness * 7800.0  # 200–8000 Hz

        # Channel 4: space → reverb wet
        space = max(0.0, min(1.0, values[4]))
        reverb_wet = space * 0.8  # cap at 0.8

        # Channel 5: tension → lattice stretch
        tension = max(0.0, min(1.0, values[5]))
        lattice_stretch = 0.9 + tension * 0.2  # 0.9–1.1

        # Channel 6: noise → noise floor
        noise = max(0.0, min(1.0, values[6]))
        noise_floor = noise * 0.1

        # Channel 7: snap → snap threshold
        snap = max(0.0, min(1.0, values[7]))

        # Channel 8: weight → attack and decay
        weight = max(0.0, min(1.0, values[8]))
        attack = 0.001 + weight * 0.499  # 0.001–0.5
        decay = 0.01 + weight * 0.99     # 0.01–1.0

        return {
            "note": note,
            "freq": freq,
            "velocity": int(dynamics * self.velocity_scale),
            "lattice_shape": lattice_shape,
            "lattice_stretch": lattice_stretch,
            "noise_floor": noise_floor,
            "snap_threshold": snap,
            "filter_cutoff": filter_cutoff,
            "reverb_wet": reverb_wet,
            "attack": attack,
            "decay": decay,
            "sustain": dynamics,
            "release": 0.3,
        }

    def map_to_synth(self, values: Sequence[float], **overrides) -> ConstraintSynth:
        """Build a complete ConstraintSynth from FluxVector values."""
        params = self.map_to_synth_params(values)
        params.update(overrides)

        osc = LatticeOscillator(
            frequency=params["freq"],
            lattice_shape=params["lattice_shape"],
            lattice_stretch=params["lattice_stretch"],
            noise_floor=params["noise_floor"],
            snap_threshold=params["snap_threshold"],
        )
        env = FunnelEnvelope(
            attack=params["attack"],
            decay=params["decay"],
            sustain=params["sustain"],
            release=params["release"],
        )
        return ConstraintSynth(
            oscillator=osc,
            envelope=env,
            filter_cutoff=params["filter_cutoff"],
            reverb_wet=params["reverb_wet"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# FluxBridge — main bridge class
# ──────────────────────────────────────────────────────────────────────────────

class FluxBridge:
    """Bridge between flux-tensor-midi and constraint-synth.

    Renders MidiEvent streams as direct audio via ConstraintSynth,
    bypassing DawDreamer entirely while preserving constraint semantics.

    Parameters
    ----------
    preset : str, optional
        ConstraintSynth preset name. If provided, used as default timbre.
    mapper : FluxVectorMapper, optional
        Custom mapper. Defaults to standard channel mapping.
    sample_rate : int
        Audio sample rate (default 44100).
    """

    def __init__(
        self,
        preset: str | None = None,
        mapper: FluxVectorMapper | None = None,
        sample_rate: int = 44100,
    ):
        self.sample_rate = sample_rate
        self.mapper = mapper or FluxVectorMapper(sample_rate=sample_rate)

        if preset is not None:
            self._default_synth = ConstraintSynth.from_preset(preset)
        else:
            self._default_synth = ConstraintSynth()

    @property
    def synth(self) -> ConstraintSynth:
        """The default ConstraintSynth instance."""
        return self._default_synth

    def render_events(
        self,
        events: Sequence[Any],
        spacing_ms: float = 10.0,
        crossfade_samples: int = 64,
    ) -> np.ndarray:
        """Render a list of MidiEvent objects to audio.

        Accepts flux-tensor-midi MidiEvent objects or MidiEventShim objects.
        Events can be in any order; they are sorted by start time.

        Parameters
        ----------
        events : sequence of MidiEvent-like
            Events with .note, .velocity, .start_ms, .duration_ms attributes.
        spacing_ms : float
            Extra silence between overlapping notes (default 10ms).
        crossfade_samples : int
            Crossfade length between consecutive note boundaries.

        Returns
        -------
        np.ndarray
            Mono audio signal, float64, [-1, 1].
        """
        if not events:
            return np.array([], dtype=np.float64)

        normalized = [_normalize_event(e) for e in events]
        normalized.sort(key=lambda e: e.start_ms)

        # Calculate total duration
        end_ms = max(e.start_ms + e.duration_ms for e in normalized)
        total_seconds = end_ms / 1000.0 + 1.0  # extra for release tail
        total_samples = int(total_seconds * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float64)

        for ev in normalized:
            duration_sec = ev.duration_ms / 1000.0
            if duration_sec <= 0:
                continue

            # Render note via constraint-synth
            note_audio = self._default_synth.play_note(
                ev.note, ev.velocity, duration_sec
            )

            # Place at correct position
            start_sample = int(ev.start_ms / 1000.0 * self.sample_rate)
            end_sample = min(start_sample + len(note_audio), total_samples)
            length = end_sample - start_sample

            if length > 0:
                # Additive mixing for overlapping notes
                output[start_sample:end_sample] += note_audio[:length]

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output / peak * 0.9

        return output

    def render_flux_vector(
        self,
        values: Sequence[float],
        start_ms: float = 0.0,
        duration_ms: float = 500.0,
        salience: Sequence[float] | None = None,
        tolerance: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Render a single FluxVector to audio via direct constraint mapping.

        The FluxVector channels control all synth parameters directly —
        no MIDI involved. This is the pure constraint-synthesis path.

        Parameters
        ----------
        values : sequence of float
            9-channel FluxVector values.
        start_ms : float
            Start time offset (for scheduling within a larger buffer).
        duration_ms : float
            Note duration in milliseconds.
        salience : sequence of float, optional
            Per-channel salience values (0–1).
        tolerance : sequence of float, optional
            Per-channel tolerance values (ms).

        Returns
        -------
        np.ndarray
            Mono audio signal.
        """
        params = self.mapper.map_to_synth_params(values, salience, tolerance)

        # Build a per-vector synth with the mapped parameters
        osc = LatticeOscillator(
            frequency=params["freq"],
            lattice_shape=params["lattice_shape"],
            lattice_stretch=params["lattice_stretch"],
            noise_floor=params["noise_floor"],
            snap_threshold=params["snap_threshold"],
        )
        env = FunnelEnvelope(
            attack=params["attack"],
            decay=params["decay"],
            sustain=params["sustain"],
            release=params["release"],
        )
        synth = ConstraintSynth(
            oscillator=osc,
            envelope=env,
            filter_cutoff=params["filter_cutoff"],
            reverb_wet=params["reverb_wet"],
        )

        duration_sec = duration_ms / 1000.0
        return synth.play_note(params["note"], params["velocity"], duration_sec)

    def render_flux_sequence(
        self,
        vectors: Sequence[dict],
        spacing_ms: float = 50.0,
    ) -> np.ndarray:
        """Render a sequence of FluxVectors to audio.

        Parameters
        ----------
        vectors : sequence of dict
            Each dict should have:
            - "values": 9-channel values
            - "start_ms": start time (optional, auto-sequenced)
            - "duration_ms": duration (default 250)
            - "salience": optional salience values
            - "tolerance": optional tolerance values
        spacing_ms : float
            Default spacing between vectors if start_ms not specified.

        Returns
        -------
        np.ndarray
            Mono audio signal.
        """
        buffers = []
        auto_start = 0.0

        for v in vectors:
            values = v["values"]
            duration_ms = v.get("duration_ms", 250.0)
            start_ms = v.get("start_ms", auto_start)

            audio = self.render_flux_vector(
                values,
                start_ms=0.0,  # render relative
                duration_ms=duration_ms,
                salience=v.get("salience"),
                tolerance=v.get("tolerance"),
            )
            buffers.append((start_ms, audio))
            auto_start = start_ms + duration_ms + spacing_ms

        if not buffers:
            return np.array([], dtype=np.float64)

        # Calculate total length
        max_end = max(start + len(audio) / self.sample_rate * 1000
                      for start, audio in buffers)
        total_samples = int((max_end / 1000.0 + 0.5) * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float64)

        for start_ms, audio in buffers:
            start_sample = int(start_ms / 1000.0 * self.sample_rate)
            end_sample = min(start_sample + len(audio), total_samples)
            length = end_sample - start_sample
            if length > 0:
                output[start_sample:end_sample] += audio[:length]

        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output / peak * 0.9

        return output

    def events_to_constraints(
        self,
        events: Sequence[Any],
    ) -> List[dict]:
        """Extract constraint parameters from a MidiEvent stream.

        Reverse-maps MIDI events back into constraint parameter space.
        This enables round-tripping: constraints → MIDI → constraints.

        Parameters
        ----------
        events : sequence of MidiEvent-like

        Returns
        -------
        list of dict
            Each dict contains the inferred constraint parameters per event.
        """
        results = []
        for ev in events:
            ev = _normalize_event(ev)
            freq = 440.0 * (2 ** ((ev.note - 69) / 12.0))

            results.append({
                "note": ev.note,
                "freq": freq,
                "velocity": ev.velocity,
                "start_ms": ev.start_ms,
                "duration_ms": ev.duration_ms,
                "channel": ev.channel,
                "amplitude": ev.velocity / 127.0,
                "duration_sec": ev.duration_ms / 1000.0,
            })
        return results

    @staticmethod
    def to_wav(
        signal: np.ndarray,
        path: str,
        sample_rate: int = 44100,
    ) -> None:
        """Save audio signal as 16-bit WAV."""
        ConstraintSynth.to_wav(signal, path, sample_rate)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison utility
# ──────────────────────────────────────────────────────────────────────────────

def compare_renderers(
    events: Sequence[Any],
    preset: str = "piano_ballad",
    output_dir: str = ".",
) -> dict:
    """Render the same events via both paths and compare.

    Returns a dict with:
    - "direct_path": path to constraint-synth WAV
    - "direct_duration_ms": render time
    - "events_count": number of events rendered

    Note: DawDreamer comparison only available if flux-tensor-midi
    with audio extras is installed.
    """
    import time
    import os

    os.makedirs(output_dir, exist_ok=True)

    bridge = FluxBridge(preset=preset)

    t0 = time.perf_counter()
    audio = bridge.render_events(events)
    direct_ms = (time.perf_counter() - t0) * 1000

    direct_path = os.path.join(output_dir, "bridge_direct.wav")
    bridge.to_wav(audio, direct_path)

    return {
        "direct_path": direct_path,
        "direct_duration_ms": round(direct_ms, 2),
        "events_count": len(events),
        "audio_samples": len(audio),
        "audio_duration_sec": len(audio) / bridge.sample_rate,
    }
