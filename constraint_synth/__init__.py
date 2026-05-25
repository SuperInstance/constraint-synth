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

# v0.5.0: Dial space, innovation cycle, perception, lattice, neural
from .dial_space import (
    DialPosition,
    TRADITIONS,
    find_cluster,
    find_nearest_tradition,
    find_unexplored,
    interpolate_traditions,
    structure_surplus,
)
from .innovation_cycle import (
    Phase,
    Style,
    WESTERN_STYLES,
    detect_phase,
    cycle_acceleration,
    predict_next_rebellion,
)
from .perception import (
    jnd,
    consonance_threshold,
    tradition_recognition,
    pleasantness,
    MOST_PLEASING,
)
from .lattice import (
    EisensteinNorm,
    LatticePoint,
    tenney_height,
    consonance_score,
    find_sangam,
    NEAREST_HARMONIC,
)
from .neural import (
    predict_fmr,
    predict_eeg,
    adaptation_rate,
    DIAL_BRAIN_CORRELATION,
)

# v0.6.0: Quality DSP effects
from .quality_effects import (
    QualityChain,
    QualityPreset,
    generate_test_signal,
    write_wav,
    read_wav,
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
    # Dial space (v0.5.0)
    "DialPosition",
    "TRADITIONS",
    "find_cluster",
    "find_nearest_tradition",
    "find_unexplored",
    "interpolate_traditions",
    "structure_surplus",
    # Innovation cycle (v0.5.0)
    "Phase",
    "Style",
    "WESTERN_STYLES",
    "detect_phase",
    "cycle_acceleration",
    "predict_next_rebellion",
    # Perception (v0.5.0)
    "jnd",
    "consonance_threshold",
    "tradition_recognition",
    "pleasantness",
    "MOST_PLEASING",
    # Lattice (v0.5.0)
    "EisensteinNorm",
    "LatticePoint",
    "tenney_height",
    "consonance_score",
    "find_sangam",
    "NEAREST_HARMONIC",
    # Neural (v0.5.0)
    "predict_fmr",
    "predict_eeg",
    "adaptation_rate",
    "DIAL_BRAIN_CORRELATION",
    # Quality DSP effects (v0.6.0)
    "QualityChain",
    "QualityPreset",
    "generate_test_signal",
    "write_wav",
    "read_wav",
]
