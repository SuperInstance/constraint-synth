# User Guide — constraint-synth

A synthesizer where **waveshape IS lattice geometry**. Instead of treating oscillator shapes as unrelated waveforms, they're all manifestations of lattice snapping — different geometries of the same underlying constraint-theory framework.

## Installation

```bash
pip install constraint-synth
```

Requires Python 3.10+, NumPy.

## Core Components

### LatticeOscillator

The oscillator maps waveform shapes to lattice geometries:

| Shape | Lattice | How it works |
|-------|---------|-------------|
| `sine` | Continuous (ε = ∞) | No snapping — pure smooth wave |
| `square` | Z₂ (binary snap) | Only 2 lattice directions: +1, -1 |
| `saw` | Z (integer lattice) | Ramp through lattice points with snap |
| `triangle` | A₂ (1D Eisenstein) | Eisenstein lattice projected to 1D |
| `eisenstein` | Full A₂ (hexagonal) | Complete hexagonal lattice tiling |

```python
from constraint_synth import LatticeOscillator

# Sine — smooth, no snapping
osc = LatticeOscillator(frequency=440.0, lattice_shape="sine")
signal = osc.generate(duration=1.0)

# Square — Z₂ binary snap
osc = LatticeOscillator(frequency=220.0, lattice_shape="square")

# Eisenstein — full hexagonal lattice
osc = LatticeOscillator(
    frequency=440.0,
    lattice_shape="eisenstein",
    lattice_stretch=1.002,  # Slight inharmonicity (piano-like)
)

# Noise floor — jitter that never converges
osc = LatticeOscillator(
    frequency=440.0,
    lattice_shape="eisenstein",
    noise_floor=0.15,  # 15% noise
)
```

### FunnelEnvelope

ADSR envelope shaped as a deadband funnel. The envelope narrows from attack to sustain, constraining the signal progressively.

```python
from constraint_synth import FunnelEnvelope

env = FunnelEnvelope(
    attack=0.005,   # 5ms attack
    decay=0.1,      # 100ms decay
    sustain=0.9,    # 90% sustain level
    release=0.15,   # 150ms release
)

# Apply to any signal
shaped = env.apply(signal, sample_rate=44100, duration=1.0)
```

### ConsonanceFilter

A harmonic filter based on consonance theory. Filters partials based on their harmonic relationship to the fundamental.

```python
from constraint_synth import ConsonanceFilter

# Warm filter — low cutoff, gentle resonance
warm = ConsonanceFilter(cutoff=0.4, resonance=1.0)

# Bright filter — high cutoff, sharp resonance
bright = ConsonanceFilter(cutoff=0.9, resonance=3.0)

# Apply
filtered = warm.apply(signal, fundamental_freq=440.0, sample_rate=44100)
```

### ConstraintSynth

The full chain: oscillator → envelope → filter.

```python
from constraint_synth import ConstraintSynth, LatticeOscillator, FunnelEnvelope, ConsonanceFilter

synth = ConstraintSynth(
    oscillator=LatticeOscillator(lattice_shape="eisenstein", lattice_stretch=1.002),
    envelope=FunnelEnvelope(attack=0.002, decay=0.4, sustain=0.3, release=0.5),
    filter=ConsonanceFilter(cutoff=0.4, resonance=1.0),
)

# Play a single note (MIDI pitch 60 = C4, velocity 100, 0.3 seconds)
signal = synth.play_note(pitch=60, velocity=100, duration=0.3)

# Render a melody: list of (pitch, velocity, duration) tuples
melody = [(60, 100, 0.3), (64, 100, 0.3), (67, 100, 0.3), (72, 100, 0.5)]
signal = synth.render_melody(melody, spacing=0.05)

# Save to WAV
ConstraintSynth.to_wav(signal, "output.wav", sample_rate=synth.oscillator.sample_rate)
```

### MIDIRenderer

Render synth output to a MIDI file (with CC automation for timbre changes):

```python
from constraint_synth import MIDIRenderer

renderer = MIDIRenderer(bpm=120)
renderer.add_note(pitch=60, velocity=100, start_beat=0.0, duration_beats=1.0)
renderer.add_note(pitch=64, velocity=90, start_beat=1.0, duration_beats=1.0)
renderer.render("output.mid")
```

## Parameter Reference

### LatticeOscillator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency` | float | 440.0 | Frequency in Hz |
| `sample_rate` | int | 44100 | Samples per second |
| `lattice_shape` | str | "sine" | sine, square, saw, triangle, eisenstein |
| `lattice_stretch` | float | 1.0 | 1.0=harmonic, >1=inharmonic |
| `noise_floor` | float | 0.0 | 0-1, jitter that never converges |
| `snap_threshold` | float | 1.0 | 0=soft snap, 1=hard snap |

### FunnelEnvelope

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attack` | float | 0.01 | Attack time in seconds |
| `decay` | float | 0.1 | Decay time in seconds |
| `sustain` | float | 0.7 | Sustain level (0-1) |
| `release` | float | 0.3 | Release time in seconds |

### ConsonanceFilter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | float | 0.5 | 0-1, harmonic cutoff |
| `resonance` | float | 1.0 | Resonance multiplier |

## Preset Recipes

### Pipe Organ (Bach-style)

```python
ConstraintSynth(
    LatticeOscillator(lattice_shape="triangle"),
    FunnelEnvelope(attack=0.005, decay=0.1, sustain=0.9, release=0.15),
    ConsonanceFilter(cutoff=0.7, resonance=1.5),
)
```

### Jazz Piano

```python
ConstraintSynth(
    LatticeOscillator(lattice_shape="eisenstein", lattice_stretch=1.002),
    FunnelEnvelope(attack=0.002, decay=0.4, sustain=0.3, release=0.5),
    ConsonanceFilter(cutoff=0.4, resonance=1.0),
)
```

### Synth Pad (Ambient)

```python
ConstraintSynth(
    LatticeOscillator(lattice_shape="sine", lattice_stretch=1.001),
    FunnelEnvelope(attack=0.8, decay=0.5, sustain=0.7, release=1.2),
    ConsonanceFilter(cutoff=0.3, resonance=0.8),
)
```

### Glitch (Experimental)

```python
ConstraintSynth(
    LatticeOscillator(lattice_shape="eisenstein", noise_floor=0.4, snap_threshold=0.8),
    FunnelEnvelope(attack=0.001, decay=0.03, sustain=0.2, release=0.01),
    ConsonanceFilter(cutoff=0.9, resonance=3.0),
)
```
