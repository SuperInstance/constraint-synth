#!/usr/bin/env python3
"""Demo pipeline — generates actual music using constraint-synth presets.

Usage:
    python3 constraint-synth/demo_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from constraint_synth.synth import ConstraintSynth


# ---------------------------------------------------------------------------
# Scale / melody helpers (standalone — no external theory package needed)
# ---------------------------------------------------------------------------

# Interval semitones from root
INTERVALS = {
    "P1": 0, "m2": 1, "M2": 2, "m3": 3, "M3": 4, "P4": 5,
    "TT": 6, "P5": 7, "m6": 8, "M6": 9, "m7": 10, "M7": 11, "P8": 12,
}

SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues":          [0, 3, 5, 6, 7, 10],
    "bebop_dominant": [0, 2, 4, 5, 7, 9, 10, 11],
    "whole_tone":     [0, 2, 4, 6, 8, 10],
}

def scale_notes(root: int, scale_name: str, octave_range: tuple[int, int] = (4, 6)) -> list[int]:
    """Build a list of MIDI pitches from a named scale over a range of octaves."""
    intervals = SCALES[scale_name]
    notes = []
    for octave in range(octave_range[0], octave_range[1] + 1):
        for iv in intervals:
            pitch = root + (octave - 4) * 12 + iv  # root is in octave 4
            notes.append(pitch)
    return notes


def random_melody(
    scale_pitches: list[int],
    n_notes: int = 16,
    duration_range: tuple[float, float] = (0.15, 0.5),
    velocity_range: tuple[int, int] = (60, 110),
    step_bias: float = 0.7,  # probability of small step vs. leap
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:
    """Generate a random melody constrained to the given scale pitches."""
    rng = rng or np.random.default_rng()
    melody = []
    idx = len(scale_pitches) // 2  # start in the middle
    for _ in range(n_notes):
        if rng.random() < step_bias:
            # small step: ±1 or ±2 scale degrees
            delta = rng.choice([-2, -1, -1, 0, 1, 1, 2])
        else:
            # leap
            delta = rng.integers(-5, 6)
        idx = max(0, min(len(scale_pitches) - 1, idx + delta))
        pitch = scale_pitches[idx]
        vel = int(rng.integers(velocity_range[0], velocity_range[1] + 1))
        dur = float(rng.uniform(duration_range[0], duration_range[1]))
        melody.append((pitch, vel, dur))
    return melody


# ---------------------------------------------------------------------------
# Demo scenes
# ---------------------------------------------------------------------------

def scene_bop_sax(rng: np.random.Generator) -> tuple[str, list]:
    """Fast bebop line in Bb mixolydian."""
    pitches = scale_notes(58, "bebop_dominant", (4, 5))
    melody = random_melody(pitches, n_notes=32, duration_range=(0.08, 0.25),
                           velocity_range=(70, 120), step_bias=0.6, rng=rng)
    return "bop_sax", melody


def scene_blues_guitar(rng: np.random.Generator) -> tuple[str, list]:
    """Slow blues lick in E minor pentatonic."""
    pitches = scale_notes(64, "blues", (3, 5))
    melody = random_melody(pitches, n_notes=16, duration_range=(0.2, 0.6),
                           velocity_range=(60, 100), step_bias=0.8, rng=rng)
    return "blues_guitar", melody


def scene_techno_bass(rng: np.random.Generator) -> tuple[str, list]:
    """Punchy bass line in A minor."""
    pitches = scale_notes(45, "minor_pentatonic", (2, 3))
    melody = random_melody(pitches, n_notes=16, duration_range=(0.1, 0.3),
                           velocity_range=(90, 127), step_bias=0.75, rng=rng)
    return "techno_bass", melody


def scene_piano_ballad(rng: np.random.Generator) -> tuple[str, list]:
    """Gentle ballad in C major."""
    pitches = scale_notes(60, "major", (4, 6))
    melody = random_melody(pitches, n_notes=20, duration_range=(0.3, 0.8),
                           velocity_range=(50, 90), step_bias=0.8, rng=rng)
    return "piano_ballad", melody


def scene_808_kick(rng: np.random.Generator) -> tuple[str, list]:
    """808 kick pattern — low sine bursts."""
    # Just repeated low notes (C1 area)
    melody = []
    for i in range(16):
        # Accent every 4th beat
        vel = 127 if i % 4 == 0 else (90 if i % 2 == 0 else 70)
        # Slight pitch variation for interest
        pitch = 24 + rng.choice([0, 0, 0, 5])
        melody.append((pitch, vel, 0.25))
    return "808_kick", melody


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(42)

    scenes = [scene_bop_sax, scene_blues_guitar, scene_techno_bass,
              scene_piano_ballad, scene_808_kick]

    all_audio = []
    sr = 44100

    for scene_fn in scenes:
        preset_name, melody = scene_fn(rng)
        synth = ConstraintSynth.from_preset(preset_name)
        sr = synth.oscillator.sample_rate
        print(f"  Rendering preset '{preset_name}' — {len(melody)} notes …")
        audio = synth.render_melody(melody, spacing=0.03, crossfade_samples=32)
        all_audio.append(audio)

    # Concatenate all scenes with a short gap between them
    gap = np.zeros(int(sr * 0.8))
    full = np.concatenate([seg if i == 0 else np.concatenate([gap, seg])
                           for i, seg in enumerate(all_audio)])

    # Normalize to avoid clipping
    peak = np.max(np.abs(full))
    if peak > 0:
        full = full / peak * 0.85

    out_path = os.path.join(out_dir, "demo_full_pipeline.wav")
    ConstraintSynth.to_wav(full, out_path, sr)
    print(f"\n✅  Full pipeline demo written to: {out_path}")
    print(f"   Duration: {len(full)/sr:.1f}s  |  Sample rate: {sr}  |  Peak: {peak:.3f}")

    # Also render individual scenes as separate files
    for scene_fn in scenes:
        rng2 = np.random.default_rng(42)  # re-seed for same melodies
        preset_name, melody = scene_fn(rng2)
        synth = ConstraintSynth.from_preset(preset_name)
        audio = synth.render_melody(melody, spacing=0.03, crossfade_samples=32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.85
        path = os.path.join(out_dir, f"demo_{preset_name}.wav")
        ConstraintSynth.to_wav(audio, path, synth.oscillator.sample_rate)
        print(f"   → demo_{preset_name}.wav  ({len(audio)/sr:.1f}s)")


if __name__ == "__main__":
    main()
