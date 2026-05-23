"""Constraint Synthesizer — where waveshape IS lattice geometry."""

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter
from .synth import ConstraintSynth

__all__ = ["LatticeOscillator", "FunnelEnvelope", "ConsonanceFilter", "ConstraintSynth"]
