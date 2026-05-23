"""Demo: 5 constraint-synth presets playing a simple melody."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_synth import (
    ConstraintSynth,
    LatticeOscillator,
    FunnelEnvelope,
    ConsonanceFilter,
)

# Simple melody: C-E-G-C (ascending arpeggio)
MELODY = [
    (60, 100, 0.3),  # C4
    (64, 100, 0.3),  # E4
    (67, 100, 0.3),  # G4
    (72, 100, 0.5),  # C5
]

PRESETS = {
    "bach_organ": ConstraintSynth(
        LatticeOscillator(lattice_shape="triangle"),
        FunnelEnvelope(attack=0.005, decay=0.1, sustain=0.9, release=0.15),
        ConsonanceFilter(cutoff=0.7, resonance=1.5),
    ),
    "joplin_piano": ConstraintSynth(
        LatticeOscillator(lattice_shape="eisenstein", lattice_stretch=1.002),
        FunnelEnvelope(attack=0.002, decay=0.4, sustain=0.3, release=0.5),
        ConsonanceFilter(cutoff=0.4, resonance=1.0),
    ),
    "debussy_pad": ConstraintSynth(
        LatticeOscillator(lattice_shape="sine", lattice_stretch=1.001),
        FunnelEnvelope(attack=0.8, decay=0.5, sustain=0.7, release=1.2),
        ConsonanceFilter(cutoff=0.3, resonance=0.8),
    ),
    "coltrane_sax": ConstraintSynth(
        LatticeOscillator(lattice_shape="saw", noise_floor=0.15),
        FunnelEnvelope(attack=0.02, decay=0.15, sustain=0.6, release=0.2),
        # No consonance filter — freedom
    ),
    "aphex_glitch": ConstraintSynth(
        LatticeOscillator(
            lattice_shape="eisenstein",
            noise_floor=0.4,
            snap_threshold=0.8,
        ),
        FunnelEnvelope(attack=0.001, decay=0.03, sustain=0.2, release=0.01),
        ConsonanceFilter(cutoff=0.9, resonance=3.0),
    ),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, synth in PRESETS.items():
        signal = synth.render_melody(MELODY, spacing=0.05)
        path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        ConstraintSynth.to_wav(signal, path, sample_rate=synth.oscillator.sample_rate)
        duration = len(signal) / synth.oscillator.sample_rate
        print(f"  ✓ {name:20s} → {path}  ({duration:.2f}s)")

    print(f"\nAll presets rendered to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
