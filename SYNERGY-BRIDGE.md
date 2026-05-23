# SYNERGY-BRIDGE.md — Unifying constraint-synth and flux-tensor-midi

## Thesis

Both projects express the same core idea: **constraint parameters drive sound**. They differ only in the output path:

| | constraint-synth | flux-tensor-midi |
|---|---|---|
| **Input** | Constraint params (lattice shape, stretch, snap, filter) | FluxVector (9-channel tensor with salience/tolerance) |
| **Intermediate** | None (direct to waveform) | MidiEvent stream → .mid file |
| **Renderer** | LatticeOscillator + FunnelEnvelope + ConsonanceFilter | DawDreamer + VST/SoundFont |
| **Philosophy** | "Waveshape IS lattice geometry" | "Tensor channels ARE note events" |

**The bridge**: FluxVectors and lattice parameters are two views of the same constraint space. A FluxVector's 9 channels can map directly to oscillator parameters, bypassing MIDI entirely.

---

## Shared Constraint Parameter Model

The unified model has two layers:

### Layer 1: FluxVector → Constraint Mapping

A `FluxVector` has 9 channels. We assign semantic meaning:

| Channel | Name | Maps to (constraint-synth) | Maps to (MIDI) |
|---|---|---|---|
| 0 | pitch | `oscillator.frequency` (via pitch scaling) | note number |
| 1 | dynamics | `envelope.sustain` + velocity | velocity |
| 2 | timbre | `oscillator.lattice_shape` index | program change |
| 3 | brightness | `filter_cutoff` | CC74 (brightness) |
| 4 | space | `reverb_wet` | CC91 (reverb) |
| 5 | tension | `oscillator.lattice_stretch` | pitch bend |
| 6 | noise | `oscillator.noise_floor` | CC1 (mod wheel) |
| 7 | snap | `oscillator.snap_threshold` | CC74 detune |
| 8 | weight | `envelope.attack` + `envelope.decay` | CC7 (volume) |

### Layer 2: Event Semantics

Both systems treat time as discrete events:
- **flux-tensor-midi**: `MidiEvent(note, velocity, start_ms, duration_ms, channel)`
- **constraint-synth**: `play_note(pitch, velocity, duration)` → numpy array

The bridge converts between these representations losslessly.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Constraint DSL     │
                    │  (FluxVector / Params)│
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────────────┐
                    │   Shared Parameter   │
                    │      Model           │
                    └──┬───────────────┬──┘
                       │               │
           ┌───────────▼──┐     ┌──────▼───────────┐
           │  Direct Path  │     │   MIDI Path       │
           │  (fast, pure)  │     │  (standard DAW)  │
           │               │     │                   │
           │ LatticeOsc    │     │ MidiEvent stream  │
           │ FunnelEnv     │     │ → .mid file       │
           │ ConsonanceFilt│     │ → DawDreamer/VST  │
           │ Biquad + Rev  │     │                   │
           └───────┬───────┘     └────────┬──────────┘
                   │                      │
                   └──────────┬───────────┘
                              │
                     ┌────────▼────────┐
                     │   Audio Output   │
                     │   (.wav / numpy) │
                     └─────────────────┘
```

---

## When to Use Which Path

### Direct Synthesis (constraint-synth)
- **Real-time**: No MIDI serialization overhead
- **Procedural**: Parameters change per-sample, not per-note
- **Constraint-native**: Lattice geometry IS the sound
- **No dependencies**: Pure numpy, no VST/SoundFont needed
- **Best for**: Experimental, generative, real-time, embedded

### MIDI Pipeline (flux-tensor-midi → DawDreamer)
- **Standard**: Output works in any DAW
- **Sampled instruments**: Realistic piano, strings, etc.
- **Human-playable**: MIDI files are interchange format
- **Production-ready**: Mix with other tracks, apply DAW effects
- **Best for**: Production music, collaboration, realistic instruments

### Hybrid (bridge)
- Use flux-tensor-midi for composition (FluxVector orchestration)
- Render via constraint-synth for speed or unique timbres
- Switch to DawDreamer for final production pass
- A/B compare both renderers from the same FluxVector score

---

## Performance Comparison

| Metric | Direct Synthesis | MIDI → DawDreamer |
|---|---|---|
| **Latency** | ~0 (per-note generation) | ~real-time (must render full MIDI) |
| **Dependencies** | numpy only | dawdreamer + VST + fluidsynth |
| **Timbral range** | Lattice shapes (sine/saw/square/tri/eisenstein) | Any VST/SoundFont instrument |
| **Realism** | Synthetic, characterful | Can be photorealistic |
| **Constraint fidelity** | Perfect (constraints ARE the sound) | Lossy (MIDI quantizes parameters) |
| **Speed (10s audio)** | ~50ms | ~2-10s (depends on VST) |
| **Portability** | Any Python environment | Requires audio plugins |

---

## Implementation: flux_bridge.py

The bridge module (`constraint_synth/flux_bridge.py`) provides:

1. **`FluxBridge`** — Converts `MidiEvent` streams to constraint-synth audio
2. **`MidiEventAdapter`** — Wraps flux-tensor-midi's `MidiEvent` for constraint-synth
3. **`FluxVectorMapper`** — Maps FluxVector channels to synth parameters
4. **`compare_renderers()`** — A/B test: same input, both output paths

### Usage

```python
from constraint_synth.flux_bridge import FluxBridge

# From flux-tensor-midi MidiEvent list
bridge = FluxBridge(preset="piano_ballad")
audio = bridge.render_events(midi_events)

# Or from a FluxVector directly
from flux_tensor_midi.core.flux import FluxVector
vec = FluxVector([0.8, 0.6, 0.0, 0.5, 0.3, 1.0, 0.0, 0.5, 0.7])
audio = bridge.render_flux_vector(vec, start_ms=0, duration_ms=500)

# Save output
bridge.to_wav(audio, "output.wav")
```

---

## Shared DSL Vision

Both projects should converge on a shared constraint DSL:

```yaml
# A constraint score that drives BOTH paths
constraints:
  - time: 0.0
    channels: [0.8, 0.6, 0.0, 0.5, 0.3, 1.0, 0.0, 0.5, 0.7]
    salience: [1.0, 0.8, 0.0, 0.7, 0.5, 1.0, 0.0, 0.6, 0.8]
    tolerance: [0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.05]
    
render:
  direct:     # constraint-synth
    preset: piano_ballad
    filter_cutoff: 3500
  midi:       # flux-tensor-midi → DawDreamer
    bpm: 120
    soundfont: /path/to/piano.sf2
```

This is the north star. The bridge is step one.
