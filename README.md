# constraint-synth — Waveshape IS Lattice Geometry

A synthesizer where oscillator shapes aren't unrelated waveforms — they're all manifestations of lattice snapping. Sine = continuous, square = Z₂ binary snap, eisenstein = A₂ hexagonal tiling. Every dial has a lattice-theoretic interpretation.

## Install

```bash
pip install constraint-synth
```

Requires Python 3.10+, NumPy.

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

## The Key Idea

Traditional synthesizers treat waveshape as a menu of unrelated options. Constraint-synth unifies them: every waveform IS a lattice geometry. The oscillator snaps continuous phase to discrete lattice directions — different lattices produce different shapes. This isn't metaphor; it's the literal signal generation mechanism.

This matters because it means every synthesizer parameter has a mathematical interpretation:
- **Waveshape** = lattice geometry (Z₂, Z, A₂)
- **Inharmonicity** = lattice stretching (Voronoi cell elongation)
- **ADSR** = deadband funnel lifecycle (convergence → pocket → divergence)
- **Filter cutoff** = consonance threshold (which lattice directions pass)
- **Noise floor** = irreducible ε-jitter

## Architecture

```
LatticeOscillator → FunnelEnvelope → ConsonanceFilter → Output (WAV/MIDI)
```

| Component | What it does | Constraint theory analogue |
|-----------|-------------|--------------------------|
| LatticeOscillator | Generate waveform via lattice snap | Snap to lattice directions |
| FunnelEnvelope | ADSR shaped as deadband funnel | Convergence → pocket → divergence |
| ConsonanceFilter | Filter by interval quality | Which lattice directions pass |

## Presets

Built-in presets demonstrating different lattice geometries:

| Preset | Shape | Character |
|--------|-------|-----------|
| Bach Organ | triangle | Tight envelope, high consonance |
| Joplin Piano | eisenstein | Piano-like stretch, moderate decay |
| Debussy Pad | sine | Long attack/release, warm |
| Coltrane Sax | saw + noise | Raw, breathy, no filter |
| Aphex Glitch | eisenstein + noise | Harsh snap, fast everything |

See `examples/demo_synth.py` for the full preset code.

## API Reference

### LatticeOscillator

```python
LatticeOscillator(
    frequency=440.0,        # Hz
    sample_rate=44100,      # samples/second
    lattice_shape="sine",   # sine | square | saw | triangle | eisenstein
    lattice_stretch=1.0,    # 1.0=harmonic, >1=inharmonic
    noise_floor=0.0,        # 0-1, jitter that never converges
    snap_threshold=1.0,     # 0=soft snap, 1=hard snap
)
```

### FunnelEnvelope

```python
FunnelEnvelope(
    attack=0.01,    # seconds
    decay=0.1,      # seconds
    sustain=0.7,    # 0-1 level
    release=0.3,    # seconds
)
```

### ConsonanceFilter

```python
ConsonanceFilter(
    cutoff=0.5,     # 0-1, harmonic cutoff
    resonance=1.0,  # rolloff sharpness
)
```

### ConstraintSynth

```python
ConstraintSynth(oscillator=None, envelope=None, filter=None)
synth.play_note(pitch=60, velocity=100, duration=0.5) → np.ndarray
synth.render_melody(melody, spacing=0.05) → np.ndarray
ConstraintSynth.to_wav(signal, path, sample_rate=44100)
```

### MIDIRenderer

```python
MIDIRenderer(bpm=120)
renderer.add_note(pitch, velocity, start_beat, duration_beats)
renderer.render(output_path)
```

## Documentation

- [User Guide](docs/USER-GUIDE.md) — Complete usage documentation
- [Developer Guide](docs/DEVELOPER-GUIDE.md) — Contributing and internals
- [Examples](examples/) — Working code with audio output

## Related

- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — The mathematical primitives underneath
- [flux-tensor-midi](https://github.com/SuperInstance/flux-tensor-midi) — 4D tensor representation of MIDI events
- [constraint-viz](https://github.com/SuperInstance/constraint-viz) — Multi-scale constraint visualization

## License

Apache 2.0
