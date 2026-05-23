# Constraint Synthesizer Prototype

A proof-of-concept synthesizer where **waveshape IS lattice geometry**.

The core idea: instead of treating oscillator shapes as unrelated waveforms, they're all manifestations of lattice snapping — different geometries of the same underlying constraint-theory framework.

## Architecture

```
LatticeOscillator → FunnelEnvelope → ConsonanceFilter → WAV
```

### LatticeOscillator
Waveshape = lattice geometry:
- **sine** — continuous manifold, no snapping
- **square** — Z₂ snap (binary: ±1)
- **saw** — ramp through Z lattice
- **triangle** — A₁ snap
- **eisenstein** — full A₂ hexagonal snap (6-level quantization)

Parameters: `lattice_stretch` (inharmonicity), `noise_floor` (irreducible jitter), `snap_threshold` (soft/hard)

### FunnelEnvelope
ADSR as deadband funnel lifecycle:
- **Attack** = convergence rate
- **Hold** = suspension plateau
- **Decay** = relaxation to pocket
- **Sustain** = equilibrium epsilon
- **Release** = divergence

### ConsonanceFilter
Filters by *interval quality* not frequency. Harmonics close to integer ratios with the fundamental pass through; dissonant partials get attenuated. `cutoff` controls the consonance threshold; `resonance` controls rolloff sharpness.

## Quick Start

```python
from constraint_synth import ConstraintSynth, LatticeOscillator, FunnelEnvelope

synth = ConstraintSynth(
    LatticeOscillator(lattice_shape="triangle", lattice_stretch=1.002),
    FunnelEnvelope(attack=0.005, decay=0.3, sustain=0.6, release=0.5),
)
signal = synth.play_note(pitch=60, velocity=100, duration=0.5)
ConstraintSynth.to_wav(signal, "note.wav")
```

## Presets

See `examples/demo_synth.py` for 5 named presets:

| Preset | Shape | Character |
|--------|-------|-----------|
| Bach Organ | triangle | Tight envelope, high consonance |
| Joplin Piano | eisenstein | Piano-like stretch, moderate decay |
| Debussy Pad | sine | Long attack/release, warm |
| Coltrane Sax | saw + noise | Raw, breathy, no filter |
| Aphex Glitch | eisenstein + noise | Harsh snap, fast everything |

## Tests

```bash
cd constraint-synth
pip install -e ".[dev]"
pytest
```

## Why This Matters

This isn't just a synth — it's a proof that constraint theory parameters map naturally to audio. Every dial on a traditional synthesizer has a lattice-theoretic interpretation:

- **Waveshape** = lattice geometry
- **Inharmonicity** = lattice stretching (Voronoi cell elongation)
- **ADSR** = deadband funnel lifecycle (convergence → pocket → divergence)
- **Filter cutoff** = consonance threshold (which lattice directions pass)
- **Noise floor** = irreducible ε-jitter

The parameter atlas (85 dials) becomes a unified mathematical framework, not a bag of unrelated knobs.
