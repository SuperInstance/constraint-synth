# constraint-synth

A constraint-theory synthesizer — generates audio governed by Eisenstein lattice constraints, deadband funnels, and conservation laws. Rust core with Python control.

## What This Gives You

- **Constraint-aware synthesis** — every tone verified against lattice and conservation constraints
- **Multi-voice engine** — polyphonic output with per-voice consonance tracking
- **Funnel dynamics** — amplitude and timbre shaped by deadband funnel decay curves
- **Conservation monitoring** — real-time spectral conservation ratio tracking
- **MIDI input** — play constraint-synthesized sounds from any MIDI controller

## Quick Start

### Python

```python
from constraint_synth import Synth, Voice

synth = Synth(sample_rate=44100, buffer_size=512)

# Add voices at lattice points
synth.add_voice(Voice(frequency=220.0, lattice=(1, 0, 0)))  # A3 fundamental
synth.add_voice(Voice(frequency=330.0, lattice=(0, 1, 0)))  # perfect fifth lattice point

# Render with constraint checking
audio = synth.render(duration=2.0)
print(f"Conservation ratio: {synth.conservation_ratio():.3f}")

# Save
synth.save("output.wav")
```

### Rust

```rust
use constraint_synth::{Synth, Voice};

let mut synth = Synth::new(44100, 512);
synth.add_voice(Voice::new(220.0, (1, 0, 0)));
synth.add_voice(Voice::new(330.0, (0, 1, 0)));

let audio = synth.render_seconds(2.0);
```

## How It Fits

The **synthesizer** built on constraint theory primitives:

- [constraint-audio](https://github.com/SuperInstance/constraint-audio) — low-level DSP this is built on
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — lattice and funnel theory
- [constraint-instrument](https://github.com/SuperInstance/constraint-instrument) — higher-level instrument API using this synth
- [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python) — conservation ratio tracking

## Testing

```bash
cargo test
```

## Installation

```bash
# Rust
cargo add constraint-synth

# Python
pip install constraint-synth
```

## License

MIT
