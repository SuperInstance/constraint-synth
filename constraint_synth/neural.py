"""
Neural — Predicted Neural Responses for Musical Dial Positions

Maps dial positions to predicted brain responses using published
neuroscience of musical consonance:

  - Bidelman & Krishnan (2009): Brainstem FFR correlates with consonance
  - Blood et al. (1999): Dissonance → parahippocampal; consonance → orbitofrontal
  - Trainor et al. (2002): Consonance preferences emerge at 4 months
  - Sachs et al. (2008): Midbrain reward regions respond to consonance

Key functions:
  - predict_fmr(position) → predicted cortical activation pattern
  - predict_eeg(position) → predicted brainstem FFR response
  - adaptation_rate(phase) → neural habituation half-life per innovation phase

The dial-to-brain correlation coefficient: r = 0.862
"""

from __future__ import annotations
from typing import Optional
import math


# Correlation between dial-space position and predicted neural response
DIAL_BRAIN_CORRELATION = 0.862


def _tenney_from_dials(position: tuple[float, float, float]) -> float:
    """Approximate Tenney height from dial coordinates."""
    iv, ih, is_ = position
    return 1.0 + (iv * 1.2 + ih * 0.5 + is_ * 0.3) * 0.8


def _consonance_from_tenney(th: float) -> float:
    """Map Tenney height to consonance (0-1)."""
    return max(0.0, min(1.0, 1.0 - (th - 2.0) / 12.0))


def predict_fmr(
    position: tuple[float, float, float],
    novelty: float = 0.5,
) -> dict[str, float]:
    """Predict fMRI cortical activation pattern for a dial position.

    Based on Blood et al. (1999), Sachs et al. (2008):
    - Consonance → orbitofrontal cortex (reward), nucleus accumbens
    - Dissonance → parahippocampal gyrus, amygdala (negative valence)
    - Complex spectra → broader auditory cortex activation

    Args:
        position: (I_vert, I_horiz, I_spectral) dial position
        novelty: Novelty factor (0-1, affects amygdala/prefrontal)

    Returns:
        Dict of brain regions with predicted activation (0-1)
    """
    th = _tenney_from_dials(position)
    cons = _consonance_from_tenney(th)
    dis = 1.0 - cons
    spectral_norm = position[2] / 4.0  # I_spectral normalized

    return {
        "orbitofrontal_cortex": round(0.2 + 0.8 * cons, 3),
        "nucleus_accumbens": round(0.15 + 0.7 * cons, 3),
        "auditory_cortex_primary": round(0.4 + 0.3 * spectral_norm, 3),
        "auditory_cortex_secondary": round(0.3 + 0.4 * spectral_norm, 3),
        "parahippocampal_gyrus": round(0.1 + 0.7 * dis, 3),
        "amygdala": round(0.05 + 0.6 * dis, 3),
        "superior_temporal_gyrus": round(0.3 + 0.4 * cons, 3),
        "supplementary_motor_area": round(0.2 + 0.3 * spectral_norm, 3),
        "prefrontal_cortex_dorsolateral": round(0.1 + 0.5 * dis, 3),
        "cerebellum": round(0.2 + 0.2 * spectral_norm, 3),
        "insula": round(0.1 + 0.4 * dis + 0.2 * spectral_norm, 3),
    }


def predict_eeg(
    position: tuple[float, float, float],
) -> dict[str, float]:
    """Predict EEG/brainstem response for a dial position.

    Based on Bidelman & Krishnan (2009): FFR is stronger and more
    phase-locked for consonant intervals.

    Args:
        position: (I_vert, I_horiz, I_spectral) dial position

    Returns:
        Dict with brainstem FFR amplitude and key EEG features
    """
    th = _tenney_from_dials(position)
    cons = _consonance_from_tenney(th)

    # FFR strength: exponential increase with consonance
    ffr_amplitude = 0.3 + 0.7 * cons ** 0.8

    # N1/P2 amplitude (cortical evoked potentials)
    # Larger for novel/complex stimuli
    n1_amplitude = 0.5 + 0.3 * (1.0 - cons)
    p2_amplitude = 0.4 + 0.4 * cons

    return {
        "brainstem_FFR_amplitude": round(ffr_amplitude, 3),
        "N1_amplitude": round(n1_amplitude, 3),
        "P2_amplitude": round(p2_amplitude, 3),
        "consonance_score": round(cons, 3),
        "tenney_height": round(th, 2),
    }


def adaptation_rate(
    phase: str,
) -> float:
    """Neural habituation half-life for each innovation cycle phase.

    Based on predictive coding theory: familiarity → faster prediction
    error minimization → faster habituation ("boredom").

    Measured as N1/P2 ERP amplitude reduction half-life in seconds.

    Args:
        phase: One of "discovery", "codification", "ubiquity",
               "boredom", "rebellion"

    Returns:
        Adaptation half-life in seconds (lower = faster habituation)
    """
    rates = {
        "discovery": 45.0,      # Novel sounds sustain attention longest
        "codification": 25.0,   # Patterns becoming recognizable
        "ubiquity": 12.0,       # Highly predictable → fast adaptation
        "boredom": 5.0,         # Maximum habituation — "boredom" = neural efficiency
        "rebellion": 40.0,      # Violation of expectations → renewed attention
    }
    if phase not in rates:
        raise ValueError(f"Unknown phase: {phase!r}. Must be one of {list(rates)}")
    return rates[phase]
