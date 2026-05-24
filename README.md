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

### High-Quality Mode (v0.2+)

```
UnisonOscillator → EnvelopeFollower → BiquadFilter → Chorus → Reverb → StereoWidth
```

| Component | What it does | Constraint theory analogue |
|-----------|-------------|--------------------------|
| LatticeOscillator | Generate waveform via lattice snap | Snap to lattice directions |
| UnisonOscillator | Multiple detuned copies for warmth | Stochastic sampling of lattice |
| FunnelEnvelope | ADSR shaped as deadband funnel | Convergence → pocket → divergence |
| ConsonanceFilter | Filter by interval quality | Which lattice directions pass |
| EnvelopeFollower | Velocity-sensitive filter modulation | ε varies with energy |
| ChorusEffect | LFO-modulated delay voices | Temporal lattice interference |
| StereoWidth | Haas-effect decorrelation | Spatial lattice dimensions |

## Presets

Built-in presets with full signal chain (unison + filter envelope + reverb + stereo):

| Preset | Shape | Unison | Character |
|--------|-------|--------|-----------|
| bop_sax | saw | 1 voice | Raw, breathy, tight filter |
| blues_guitar | square | 1 voice | Gritty, pitch transient, warm reverb |
| techno_bass | saw | 3 voices | Sub-heavy, resonant filter sweep |
| piano_ballad | triangle | 2 voices | Detuned warmth, long reverb |
| 808_kick | sine | 1 voice | Pitch drop, tight envelope |

```python
synth = ConstraintSynth.from_preset("piano_ballad")
audio = synth.play_note(pitch=60, velocity=100, duration=0.5)
```

## Play-Along Mode (v0.2+)

AI reacts to your playing in real-time — analyzes key, tempo, density and generates scale-aware responses.

```python
from constraint_synth import PlayAlong, PlayAlongConfig, ResponseStrategy

pa = PlayAlong(PlayAlongConfig(
    key="C", mode="major",
    strategy=ResponseStrategy.COMPLEMENT,
    creativity=0.3,
))

# Feed your notes
pa.feed(note=60, velocity=100, timestamp_ms=0)
pa.feed(note=64, velocity=90, timestamp_ms=500)
pa.feed(note=67, velocity=95, timestamp_ms=1000)

# Get AI response
responses = pa.respond()
for r in responses:
    print(f"{r.note_name} vel={r.velocity} @ {r.start_ms:.0f}ms")

# Render to audio
audio = pa.render_response(responses, preset="piano_ballad")
ConstraintSynth.to_wav(audio, "response.wav")
```

### Strategies

| Strategy | Description |
|----------|------------|
| COMPLEMENT | Fills gaps with scale tones that complement the harmony |
| COUNTERPOINT | Contrary motion — moves opposite to input melody |
| ECHO | Delayed repetition with pitch/timing variation |
| BASS | Root motion and bass line from harmonic context |
| CHORDAL | Block chords (triads) on strong beats |
| FREE | Constraint-guided improvisation with creativity parameter |

### Auto Mode

```python
# Auto-detect key and strategy from your playing
pa = PlayAlong(PlayAlongConfig(
    key="auto", mode="auto",
    creativity=0.4,
))
```

Uses Krumhansl-Schmuckler key detection and auto-selects strategy based on your playing density and tempo.

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
ConstraintSynth(oscillator=None, envelope=None, filter=None,
                 filter_cutoff=2000.0, reverb_wet=0.3,
                 pitch_envelope=None, filter_envelope=None,
                 velocity_sensitivity=0.7,
                 unison_voices=1, unison_detune_cents=0.0,
                 stereo_width=0.0)
synth.play_note(pitch=60, velocity=100, duration=0.5) → np.ndarray
synth.render_melody(melody, spacing=0.05) → np.ndarray
ConstraintSynth.to_wav(signal, path, sample_rate=44100)
```

### PlayAlong

```python
PlayAlong(PlayAlongConfig(
    key="auto",              # "auto" or e.g. "C"
    mode="auto",             # "auto" or scale name
    strategy=ResponseStrategy.COMPLEMENT,
    response_delay_ms=200.0,
    creativity=0.3,          # 0=safe, 1=wild
    octave_offset=-1,        # response below input
    max_response_notes=4,
))
pa.feed(note=60, velocity=100, timestamp_ms=0)
pa.respond() → List[ResponseEvent]
pa.render_response(responses) → np.ndarray
pa.get_status() → dict
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
