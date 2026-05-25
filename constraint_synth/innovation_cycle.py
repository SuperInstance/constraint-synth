"""
Innovation Cycle — How Artistic Styles Emerge, Spread, and Die

Models the six-phase cycle of artistic innovation:
  1. DISCOVERY    — innovator finds new dial position
  2. CODIFICATION — academics extract rules from the innovation
  3. UBIQUITY     — technology amplifies the style to default status
  4. BOREDOM      — next generation finds it tired
  5. REBELLION    — new generation breaks the rules (finds new position)
  6. → Discovery  — cycle restarts

The cycle accelerates over time because reproductive technology
compresses each phase transition.

References:
  INNOVATION-CYCLE.md — the full formal model with testable predictions
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class Phase(Enum):
    """The six phases of the Innovation Cycle."""
    DISCOVERY = "discovery"
    CODIFICATION = "codification"
    UBIQUITY = "ubiquity"
    BOREDOM = "boredom"
    REBELLION = "rebellion"


@dataclass(frozen=True)
class Style:
    """A musical style with its dial position and historical dates."""
    name: str
    dial_position: tuple[float, float, float]  # (I_vert, I_horiz, I_spectral)
    year_start: int
    year_end: int
    phase: Phase


# ── Historical Western Styles ─────────────────────────────────────────────

WESTERN_STYLES: list[Style] = [
    Style("Renaissance",          (1.80, 1.20, 0.8), 1400, 1600, Phase.BOREDOM),
    Style("Baroque",              (2.20, 1.50, 1.0), 1600, 1750, Phase.BOREDOM),
    Style("Classical",            (2.00, 1.80, 1.1), 1750, 1830, Phase.BOREDOM),
    Style("Romantic",             (2.60, 2.20, 1.3), 1830, 1900, Phase.BOREDOM),
    Style("Ragtime",              (2.30, 2.80, 1.4), 1899, 1920, Phase.BOREDOM),
    Style("Early Jazz",           (2.50, 2.80, 1.5), 1920, 1942, Phase.BOREDOM),
    Style("Bebop",                (2.80, 3.20, 1.6), 1942, 1955, Phase.BOREDOM),
    Style("Rock and Roll",        (2.30, 2.50, 2.0), 1955, 1970, Phase.BOREDOM),
    Style("Minimalism",           (1.50, 2.80, 2.2), 1970, 1990, Phase.CODIFICATION),
    Style("Electronic",           (2.00, 3.00, 2.8), 1995, 2015, Phase.UBIQUITY),
    Style("Hip-hop",              (2.50, 3.50, 2.5), 1979, 2020, Phase.BOREDOM),
    Style("AI-generated",         (3.00, 3.00, 3.0), 2023, 2026, Phase.DISCOVERY),
]


def detect_phase(
    style: Style,
    metrics: Optional[dict[str, bool]] = None,
) -> Phase:
    """Detect which phase a style is in based on measurable indicators.

    Uses the Phase Detection Algorithm from INNOVATION-CYCLE.md §2:
    - d_NN: nearest-neighbor distance to known tradition clusters
    - C: codification (published pedagogy exists?)
    - U: ubiquity (commercial use by non-specialists?)
    - S_school: taught in formal curricula?

    Args:
        style: The style to classify
        metrics: Dict with optional keys 'codified', 'ubiquitous', 'in_school'

    Returns:
        Detected Phase
    """
    if metrics is None:
        # Fall back to the style's own declared phase
        return style.phase

    codified = metrics.get("codified", False)
    ubiquitous = metrics.get("ubiquitous", False)
    in_school = metrics.get("in_school", False)

    # Phase detection algorithm (simplified)
    if in_school:
        return Phase.BOREDOM
    if ubiquitous:
        return Phase.UBIQUITY
    if codified:
        return Phase.CODIFICATION

    # Check if it's a rebellion (far from previous style cluster)
    from constraint_synth.dial_space import DialPosition, TRADITIONS
    pos = DialPosition(*style.dial_position)
    nearest_name, nearest_dist = find_nearest_tradition_simple(pos)

    if nearest_dist > 1.0:
        # Far from any tradition — likely rebellion or discovery
        if style.phase == Phase.REBELLION:
            return Phase.REBELLION
        return Phase.DISCOVERY

    return Phase.DISCOVERY


def find_nearest_tradition_simple(position) -> tuple[str, float]:
    """Simple nearest-tradition lookup without full import overhead."""
    from constraint_synth.dial_space import TRADITIONS
    best_name = ""
    best_dist = float("inf")
    iv, ih, is_ = position.I_vert, position.I_horiz, position.I_spectral
    for name, pos in TRADITIONS.items():
        d = math.sqrt(
            (iv - pos.I_vert) ** 2 +
            (ih - pos.I_horiz) ** 2 +
            (is_ - pos.I_spectral) ** 2
        )
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def cycle_acceleration() -> list[tuple[str, int, int]]:
    """Compute cycle times for each historical transition.

    Returns list of (style_name, year_start, cycle_time_years).
    Cycle time = years from Discovery to Ubiquity.

    The observed trend: cycle time halves approximately every century.
    """
    transitions = [
        ("Renaissance", 1400, 180),
        ("Baroque", 1600, 150),
        ("Classical", 1750, 80),
        ("Romantic", 1830, 70),
        ("Ragtime", 1899, 43),
        ("Early Jazz", 1920, 22),
        ("Bebop", 1942, 13),
        ("Rock and Roll", 1955, 15),
        ("Hip-hop", 1979, 25),
        ("Electronic", 1995, 20),
        ("AI-generated", 2023, 10),
    ]
    return transitions


def predict_next_rebellion(current_year: int) -> dict[str, float]:
    """Predict when the next rebellion phase will occur.

    Uses exponential decay model: T_cycle(t) ≈ T₀ · 2^(-t/τ)
    where T₀ ≈ 200 years, τ ≈ 100 years.

    Args:
        current_year: The reference year

    Returns:
        Dict with predicted year, cycle time, and model parameters.
    """
    T0 = 200.0  # Base cycle time (Renaissance)
    tau = 100.0  # Halving time in years
    reference_year = 1400.0

    years_from_ref = current_year - reference_year
    predicted_cycle = T0 * 2.0 ** (-years_from_ref / tau)

    # Minimum cycle time: 2 years (human perception + cultural propagation)
    predicted_cycle = max(predicted_cycle, 2.0)

    rebellion_year = current_year + predicted_cycle * 0.8  # rebellion ≈ 80% through cycle

    return {
        "current_year": current_year,
        "predicted_cycle_years": round(predicted_cycle, 1),
        "predicted_rebellion_year": round(rebellion_year),
        "model": "T(t) = 200 × 2^(-t/100)",
        "halving_time_years": tau,
    }
