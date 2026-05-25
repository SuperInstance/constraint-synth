"""
Dial Space — Musical Traditions as Points in Parameter Space

Models musical styles as coordinates (I_vert, I_horiz, I_spectral) where:
  I_vert     = vertical information (harmonic complexity, consonance)
  I_horiz    = horizontal information (rhythmic complexity, temporal structure)
  I_spectral = spectral information (timbral richness, overtone content)

Based on the "dials not laws" insight: musical traditions are parameter settings,
not rule systems. Two traditions at similar dial positions will sound similar
regardless of geographic distance.

References:
  DIALS-NOT-LAWS.md — the parameter-space framework
  dial_space.py — original cluster analysis and audio synthesis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class DialPosition:
    """A point in (I_vert, I_horiz, I_spectral) parameter space."""
    I_vert: float
    I_horiz: float
    I_spectral: float

    def distance_to(self, other: DialPosition) -> float:
        """Euclidean distance between two dial positions."""
        return math.sqrt(
            (self.I_vert - other.I_vert) ** 2 +
            (self.I_horiz - other.I_horiz) ** 2 +
            (self.I_spectral - other.I_spectral) ** 2
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.I_vert, self.I_horiz, self.I_spectral)


# ── Tradition Data ────────────────────────────────────────────────────────

TRADITIONS: dict[str, DialPosition] = {
    "Hindustani":   DialPosition(2.77, 3.45, 2.5),
    "Carnatic":     DialPosition(2.77, 3.63, 2.8),
    "Arabic":       DialPosition(2.94, 3.10, 2.3),
    "Turkish":      DialPosition(2.83, 3.28, 2.2),
    "Javanese":     DialPosition(2.31, 2.75, 3.0),
    "Balinese":     DialPosition(2.31, 3.10, 3.2),
    "Gagaku":       DialPosition(2.38, 1.70, 3.5),
    "Chinese":      DialPosition(2.32, 2.05, 2.0),
    "West African": DialPosition(2.41, 3.63, 2.6),
    "Western ET":   DialPosition(2.72, 2.05, 1.8),
}


def find_cluster(
    position: DialPosition,
    traditions: Optional[dict[str, DialPosition]] = None,
) -> list[tuple[str, float]]:
    """Find traditions nearest to a given dial position.

    Returns list of (tradition_name, distance) sorted by proximity.
    """
    if traditions is None:
        traditions = TRADITIONS
    distances = [
        (name, position.distance_to(pos))
        for name, pos in traditions.items()
    ]
    distances.sort(key=lambda x: x[1])
    return distances


def find_nearest_tradition(
    position: DialPosition,
    traditions: Optional[dict[str, DialPosition]] = None,
) -> tuple[str, float]:
    """Find the single nearest tradition to a dial position.

    Returns (tradition_name, distance).
    """
    cluster = find_cluster(position, traditions)
    return cluster[0]


def find_unexplored(
    positions: Optional[list[DialPosition]] = None,
    bounds: tuple[float, float] = (1.0, 4.0),
    n_grid: int = 20,
) -> list[DialPosition]:
    """Find dial positions far from all known traditions.

    Grid-searches the parameter space and returns the positions
    that are farthest from any known tradition point.

    Args:
        positions: Known positions (defaults to all traditions)
        bounds: (min, max) for each axis
        n_grid: Grid resolution per axis

    Returns:
        List of DialPositions sorted by isolation (most isolated first).
    """
    if positions is None:
        positions = list(TRADITIONS.values())

    lo, hi = bounds
    step = (hi - lo) / n_grid
    results: list[tuple[DialPosition, float]] = []

    for iv_i in range(n_grid + 1):
        for ih_i in range(n_grid + 1):
            for is_i in range(n_grid + 1):
                pt = DialPosition(
                    lo + iv_i * step,
                    lo + ih_i * step,
                    lo + is_i * step,
                )
                min_dist = min(pt.distance_to(p) for p in positions)
                results.append((pt, min_dist))

    results.sort(key=lambda x: x[1], reverse=True)
    return [pt for pt, _ in results[:20]]


def interpolate_traditions(
    t1: DialPosition,
    t2: DialPosition,
    alpha: float,
) -> DialPosition:
    """Linearly interpolate between two dial positions.

    Args:
        t1: First position
        t2: Second position
        alpha: Interpolation factor (0.0 = t1, 1.0 = t2)

    Returns:
        Interpolated DialPosition
    """
    return DialPosition(
        t1.I_vert * (1 - alpha) + t2.I_vert * alpha,
        t1.I_horiz * (1 - alpha) + t2.I_horiz * alpha,
        t1.I_spectral * (1 - alpha) + t2.I_spectral * alpha,
    )


def structure_surplus(
    position: DialPosition,
    traditions: Optional[dict[str, DialPosition]] = None,
) -> float:
    """Compute structure surplus at a dial position.

    Structure surplus measures how much coherent musical structure a
    dial position can produce above random baseline. Approximated by
    proximity-weighted consonance of nearby traditions.

    Higher values indicate positions that support rich, coherent music.

    Args:
        position: The dial position to evaluate
        traditions: Known traditions for reference

    Returns:
        Structure surplus score (higher = more coherent music possible)
    """
    if traditions is None:
        traditions = TRADITIONS

    cluster = find_cluster(position, traditions)
    # Weighted sum: closer traditions contribute more
    surplus = 0.0
    total_weight = 0.0
    for name, dist in cluster:
        # Gaussian kernel weighting
        weight = math.exp(-(dist ** 2) / 1.0)
        # Traditions with lower I_vert (simpler harmony) have higher
        # intrinsic consonance, contributing to structure
        trad = traditions[name]
        consonance = 1.0 / (1.0 + (trad.I_vert - 2.0) ** 2)
        surplus += weight * consonance
        total_weight += weight

    if total_weight > 0:
        surplus /= total_weight

    # Add bonus for being in a well-explored region (near multiple traditions)
    nearby = sum(1 for _, d in cluster if d < 1.0)
    density_bonus = min(nearby / 5.0, 1.0) * 0.2

    return surplus + density_bonus
