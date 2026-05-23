# Developer Guide — constraint-synth

## Architecture

```
constraint_synth/
├── __init__.py           # Public API exports
├── oscillator.py         # LatticeOscillator — waveshape as lattice geometry
├── envelope.py           # FunnelEnvelope — ADSR shaped as deadband funnel
├── constraint_filter.py  # ConsonanceFilter — harmonic filtering
├── synth.py              # ConstraintSynth — full chain + WAV output
├── playback.py           # AudioPlayer — real-time playback
└── midi_renderer.py      # MIDIRenderer — MIDI file output
```

### Signal Flow

```
LatticeOscillator → FunnelEnvelope → ConsonanceFilter → Output
     (generate)       (apply)           (apply)         (WAV/MIDI)
```

### Key Design Decisions

1. **Lattice shapes ARE waveforms.** The `lattice_shape` parameter isn't a metaphor — it literally determines which lattice geometry the oscillator snaps to. Sine = no lattice (continuous), square = Z₂ (binary), eisenstein = A₂ (hexagonal).

2. **Stretch factor for inharmonicity.** `lattice_stretch` detunes harmonics from perfect integer ratios. 1.002 gives piano-like inharmonicity; 1.5 gives bell-like tones.

3. **Noise floor models imperfection.** Real instruments never perfectly converge. `noise_floor` adds controlled jitter that prevents the signal from ever fully settling.

## Contributing

### Setup

```bash
git clone https://github.com/SuperInstance/constraint-synth.git
cd constraint-synth
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Adding a New Lattice Shape

1. Add the shape to the `Literal` type in `oscillator.py`
2. Implement the generation logic in `LatticeOscillator.generate()`
3. Add a test in `tests/test_synth.py`
4. Update the docs

### Code Style

- Type hints on all public methods
- Docstrings on all public classes and functions
- NumPy for all numerical work
- Dataclasses for configuration objects
