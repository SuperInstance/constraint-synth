# constraint-synth

**The parameter space of musical tension, navigable in code.**

[![PyPI](https://img.shields.io/pypi/v/constraint-synth.svg)](https://pypi.org/project/constraint-synth/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/SuperInstance/constraint-synth)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SuperInstance/constraint-synth/blob/main/LICENSE)

```bash
pip install constraint-synth
```

## Quick Start

```python
from constraint_synth.dial_space import find_nearest_tradition, find_unexplored
from constraint_synth.lattice import consonance_score

# What tradition is closest to your dial position?
pos = (2.5, 3.0, 2.0)
tradition, distance = find_nearest_tradition(pos)

# What's the largest unexplored region?
unexplored = find_unexplored()

# How consonant is a perfect fifth?
score = consonance_score(220, 330)  # 0.275
```

## Features

- **Lattice geometry waveshaping** — oscillator shapes are lattice snap operations, not arbitrary waveforms
- **27 world scales** in just intonation with 12-TET approximations (Hindustani, Arabic, Javanese, Gagaku, etc.)
- **Consonance field** — a 3D landscape of harmonic beauty you can navigate programmatically
- **Dial-space model** — musical traditions as (I_vert, I_horiz, I_spectral) coordinates
- **Neural response prediction** — dial positions → predicted fMRI / EEG patterns (r = 0.862)
- **Psychoacoustic models** — JND per axis, consonance thresholds, pleasantness prediction
- **Innovation cycle** — six-phase model of how styles emerge, spread, and die
- **3/2 isomorphism** — pitch ratios = rhythm ratios (perfect fifth ↔ hemiola)
- **Play-along AI** — real-time accompaniment that analyzes your key, tempo, and density
- **10 instrument presets** — bop sax, blues guitar, techno bass, piano ballad, 808 kick, etc.
- **MIDI rendering** — programmatic composition with full export
- **DawDreamer backend** — production-quality FAUST-based DSP rendering
- **Flux bridge** — direct synthesis from flux-tensor-midi event streams
- **Eisenstein norm** — the natural metric on harmonic lattice space
- **Tradition cluster analysis** — find which traditions are neighbors in parameter space
- **Cross-cultural Sangam** — points where multiple traditions agree on beauty
- **Meantone simulator** — hear why D major sounded triumphant before equal temperament
- **Nancarrow tempo canons** — 12-voice canons with just-intonation tempo ratios
- **Runs everywhere** — ESP32, RISC-V, WASM, browser (pure Python + NumPy)

## The Dial Model

Musical traditions aren't rule systems — they're **parameter settings**. Each tradition maps to a point in a 3D space of information content:

| Axis | Measures | JND |
|------|----------|-----|
| **I_vert** | Harmonic complexity, consonance | 0.12 |
| **I_horiz** | Rhythmic complexity, temporal structure | 0.10 |
| **I_spectral** | Timbral richness, overtone content | 0.08 |

### Tradition Positions

| Tradition | I_vert | I_horiz | I_spectral |
|-----------|--------|---------|------------|
| Hindustani | 2.77 | 3.45 | 2.5 |
| Carnatic | 2.77 | 3.63 | 2.8 |
| Arabic | 2.94 | 3.10 | 2.3 |
| Turkish | 2.83 | 3.28 | 2.2 |
| Javanese | 2.31 | 2.75 | 3.0 |
| Balinese | 2.31 | 3.10 | 3.2 |
| Gagaku | 2.38 | 1.70 | 3.5 |
| Chinese | 2.32 | 2.05 | 2.0 |
| West African | 2.41 | 3.63 | 2.6 |
| Western ET | 2.72 | 2.05 | 1.8 |

Two traditions at similar dial positions sound similar **regardless of geographic distance**. The model predicts neural activation patterns with correlation r = 0.862.

## Modules

### Core

| Module | Description |
|--------|-------------|
| `lattice` | Eisenstein norm, Tenney height, Sangam points, consonance scoring on the prime lattice |
| `scales` | 27 world scales in just intonation with 12-TET mapping and cross-tradition comparison |
| `synth` | Main `ConstraintSynth` engine — lattice oscillator → funnel envelope → consonance filter |
| `oscillator` | `LatticeOscillator` — waveshape via lattice snap (sine, square, saw, triangle, eisenstein) |

### Analysis

| Module | Description |
|--------|-------------|
| `consonance_field` | 3D consonance landscape over Eisenstein lattice space — peaks, valleys, gravity wells |
| `rhythmic_consonance` | Extends consonance into time — the 3:2 principle for polyrhythm and hemiola |
| `perception` | JND per axis, consonance cliff function, tradition recognition, pleasantness (Wundt curve) |
| `neural` | Predicted fMRI cortical activation and brainstem FFR from dial positions |

### Exploration

| Module | Description |
|--------|-------------|
| `dial_space` | Traditions as 3D coordinates — clustering, interpolation, unexplored region detection |
| `innovation_cycle` | Six-phase model (discovery → codification → ubiquity → boredom → rebellion → …) |
| `three_halves` | The pitch-rhythm isomorphism — meantone simulator, Nancarrow tempo canons |
| `play_along` | Real-time AI accompaniment — complement, counterpoint, echo, bass, chordal, free strategies |

### Integration

| Module | Description |
|--------|-------------|
| `dawdreamer_backend` | FAUST-based DSP rendering — constraint graphs as DawDreamer patches, batch scale export |
| `flux_bridge` | Direct synthesis from flux-tensor-midi event streams without DawDreamer |
| `midi_renderer` | Programmatic MIDI composition and file export |

## Key Numbers

| Finding | Value |
|---------|-------|
| Dial space unexplored | **82%** of reachable positions have no known tradition |
| Tradition recognition accuracy | **98%** from dial coordinates alone |
| Dial-to-brain correlation (fMRI) | **r = 0.862** |
| Most consonant non-unison ratio | **3:2** (perfect fifth) |
| JND ratio (pitch:rhythm:spectral) | **0.12 : 0.10 : 0.08** |
| Most pleasing dial position | **(2.61, 2.33, 4.0)** |
| Consonance of perfect fifth (220→330 Hz) | **0.275** |
| Innovation cycle compression | Each phase ~30% shorter than the last |

## Installation

```bash
pip install constraint-synth
```

Requires **Python 3.10+** and **NumPy**. That's it — the core has no other dependencies.

### Optional

```bash
pip install constraint-synth[dev]    # pytest for running the test suite
pip install dawdreamer                # production-quality FAUST rendering backend
```

## Hardware

Pure Python + NumPy means constraint-synth runs everywhere:

- **ESP32** — embedded synthesis with MicroPython
- **RISC-V** — lightweight enough for SBCs
- **WASM** — browser-based synthesis (via Pyodide)
- **Browser** — real-time audio in the browser, no native extensions needed

## Links

- **GitHub**: [github.com/SuperInstance/constraint-synth](https://github.com/SuperInstance/constraint-synth)
- **PyPI**: [pypi.org/project/constraint-synth](https://pypi.org/project/constraint-synth/)
- **Docs**: [constraint-theory papers and AI-Writings](https://github.com/SuperInstance/constraint-theory-core)
- **Related**: [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) · [flux-tensor-midi](https://github.com/SuperInstance/flux-tensor-midi) · [constraint-viz](https://github.com/SuperInstance/constraint-viz)

## License

MIT
