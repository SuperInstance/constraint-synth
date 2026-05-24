"""Constraint Synthesizer — where waveshape IS lattice geometry."""

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter
from .synth import ConstraintSynth, BiquadLowpass, SchroederReverb
from .playback import AudioPlayer
from .midi_renderer import MIDIRenderer
from .sound_engine import (
    SoundEngine,
    UnisonOscillator,
    StereoWidth,
    ChorusEffect,
    EnvelopeFollower,
    build_sound_engine,
    HIGH_QUALITY_PRESETS,
)
from .dawdreamer_backend import (
    AudioBackend,
    NumpyBackend,
    DawDreamerBackend,
    FAUSTGenerator,
    ConstraintGraph,
    ConstraintNode,
    MIDIConstraintConfig,
    MIDIConstraintTransformer,
    create_backend,
    render_scale,
    render_all_scales,
    generate_faust_scales,
)

__all__ = [
    "LatticeOscillator",
    "FunnelEnvelope",
    "ConsonanceFilter",
    "ConstraintSynth",
    "AudioPlayer",
    "MIDIRenderer",
    "BiquadLowpass",
    "SchroederReverb",
    "SoundEngine",
    "UnisonOscillator",
    "StereoWidth",
    "ChorusEffect",
    "EnvelopeFollower",
    "build_sound_engine",
    "HIGH_QUALITY_PRESETS",
    # DawDreamer backend
    "AudioBackend",
    "NumpyBackend",
    "DawDreamerBackend",
    "FAUSTGenerator",
    "ConstraintGraph",
    "ConstraintNode",
    "MIDIConstraintConfig",
    "MIDIConstraintTransformer",
    "create_backend",
    "render_scale",
    "render_all_scales",
    "generate_faust_scales",
]
