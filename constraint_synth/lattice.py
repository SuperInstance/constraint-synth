"""
Lattice — Core Lattice Mathematics for Constraint Synthesis

The fundamental insight: musical ratios live on a 2^a × 3^b × 5^c lattice.
Consonance is proximity on this lattice. This module provides the core
mathematics that underpin the entire constraint synthesis framework.

Key concepts:
  - EisensteinNorm: the natural metric on harmonic lattice space
  - LatticePoint: position on the 2^a × 3^b × 5^c lattice
  - Tenney height: logarithmic complexity of a ratio
  - Sangam: points where multiple traditions agree (universal consonance)

References:
  Tenney (1983): John Cage and the Theory of Harmony
  Euler (1739): Tentamen novae theoriae musicae
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional
import math


# ── Eisenstein Norm ───────────────────────────────────────────────────────

class EisensteinNorm:
    """The natural metric on harmonic lattice space.

    The Eisenstein norm measures the "distance" of a musical ratio from
    unison (1/1) on the prime lattice. It accounts for the fact that
    prime 3 (perfect fifth) and prime 5 (major third) have different
    perceptual weights than prime 2 (octave).

    For a ratio p/q, the Eisenstein norm considers the weighted
    contributions of each prime factor.
    """

    # Weights for each prime dimension
    PRIME_WEIGHTS = {2: 1.0, 3: 1.5, 5: 2.0, 7: 2.5, 11: 3.0, 13: 3.5}

    @classmethod
    def norm(cls, ratio: Fraction) -> float:
        """Compute Eisenstein norm of a ratio.

        Lower = more consonant. The norm weights primes by their
        perceptual "distance" from unity.

        Args:
            ratio: A just-intonation ratio as a Fraction

        Returns:
            Eisenstein norm (float, lower = more consonant)
        """
        f = Fraction(ratio)
        product = f.numerator * f.denominator

        norm_val = 0.0
        n = product
        p = 2
        while p * p <= n:
            exp = 0
            while n % p == 0:
                exp += 1
                n //= p
            weight = cls.PRIME_WEIGHTS.get(p, float(math.sqrt(p)))
            norm_val += exp * weight
            p += 1
        if n > 1:
            weight = cls.PRIME_WEIGHTS.get(n, float(math.sqrt(n)))
            norm_val += weight

        return norm_val

    @classmethod
    def distance(cls, ratio1: Fraction, ratio2: Fraction) -> float:
        """Eisenstein distance between two ratios."""
        return cls.norm(ratio1 / ratio2)


# ── Lattice Point ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatticePoint:
    """A position on the 2^a × 3^b × 5^c harmonic lattice.

    Every just-intonation ratio can be expressed as 2^a × 3^b × 5^c
    (up to 5-limit; extend to 7-limit by adding another coordinate).

    Attributes:
        a: Power of 2 (octave dimension)
        b: Power of 3 (fifth dimension)
        c: Power of 5 (third dimension)
    """
    a: int
    b: int
    c: int

    @property
    def ratio(self) -> Fraction:
        """The musical ratio this lattice point represents."""
        return Fraction(2 ** self.a * 3 ** self.b * 5 ** self.c)

    @property
    def frequency_ratio(self) -> float:
        """The ratio as a float."""
        return float(self.ratio)

    def tenney_height(self) -> float:
        """Tenney height: log₂(n) + log₂(d) for ratio n/d."""
        f = self.ratio
        if f.numerator == 0:
            return float('inf')
        return math.log2(f.numerator) + math.log2(f.denominator)

    def consonance(self) -> float:
        """Consonance score (0-1, higher = more consonant)."""
        th = self.tenney_height()
        return math.exp(-0.5 * th)

    def lattice_distance(self, other: LatticePoint) -> float:
        """Manhattan distance on the prime lattice."""
        return abs(self.a - other.a) + abs(self.b - other.b) + abs(self.c - other.c)

    def normalized_cents(self) -> float:
        """Interval in cents, normalized to one octave."""
        r = self.ratio
        while r >= 2:
            r = r / 2
        while r < 1:
            r = r * 2
        return 1200.0 * math.log2(float(r))


# ── Core Functions ────────────────────────────────────────────────────────

def tenney_height(ratio: Fraction) -> float:
    """Tenney height: log₂(n) + log₂(d) for simplified fraction n/d.

    The standard consonance metric from Tenney (1983).
    Lower = more consonant. Unison = 0, octave = 1.

    Args:
        ratio: A just-intonation ratio

    Returns:
        Tenney height (float)
    """
    f = Fraction(ratio)
    if f.numerator == 0:
        return float('inf')
    return math.log2(f.numerator) + math.log2(f.denominator)


def consonance_score(freq1: float, freq2: float) -> float:
    """Consonance between two frequencies.

    Finds the simplest ratio approximation of the interval and
    computes consonance via inverse Tenney height.

    Args:
        freq1: First frequency in Hz
        freq2: Second frequency in Hz

    Returns:
        Consonance score (0-1, higher = more consonant)
    """
    if freq1 <= 0 or freq2 <= 0:
        return 0.0

    ratio = freq2 / freq1

    # Normalize to one octave
    while ratio >= 2.0:
        ratio /= 2.0
    while ratio < 1.0:
        ratio *= 2.0

    # Find best simple ratio approximation
    best_consonance = 0.0
    for denom in range(1, 100):
        numer = round(ratio * denom)
        if numer < 1 or numer > 200:
            continue
        actual = numer / denom
        cents_diff = abs(1200.0 * math.log2(actual / ratio))
        if cents_diff < 15:  # within 15 cents
            f = Fraction(numer, denom)
            th = tenney_height(f)
            c = math.exp(-0.5 * th)
            if c > best_consonance:
                best_consonance = c

    # Also check exact simple ratios directly
    for numer in range(1, 32):
        for denom in range(1, 32):
            actual_ratio = numer / denom
            actual_cents = 1200.0 * math.log2(actual_ratio)
            # Normalize to one octave
            while actual_cents > 1200:
                actual_cents -= 1200
            while actual_cents < 0:
                actual_cents += 1200
            if abs(actual_cents - 1200.0 * math.log2(ratio)) < 15:
                f = Fraction(numer, denom)
                th = tenney_height(f)
                c = math.exp(-0.5 * th)
                if c > best_consonance:
                    best_consonance = c

    return best_consonance


def find_sangam(
    traditions: dict[str, list[Fraction]],
    tolerance_cents: float = 50.0,
) -> list[Fraction]:
    """Find universal consonance points (sangam) across traditions.

    A sangam is a ratio that appears in multiple traditions' scale systems.
    These represent universal consonance — intervals that humans across
    cultures independently converged upon.

    Args:
        traditions: Dict mapping tradition name to list of scale ratios
        tolerance_cents: How close two ratios must be to count as "same"

    Returns:
        List of Fraction ratios that appear in 3+ traditions
    """
    from collections import Counter

    # Flatten all ratios with tradition tags
    ratio_traditions: dict[Fraction, set[str]] = {}
    for trad_name, ratios in traditions.items():
        for r in ratios:
            r = Fraction(r)
            matched = False
            for existing in ratio_traditions:
                cents_diff = abs(1200.0 * math.log2(float(r) / float(existing)))
                if cents_diff < tolerance_cents:
                    ratio_traditions[existing].add(trad_name)
                    matched = True
                    break
            if not matched:
                ratio_traditions[r] = {trad_name}

    # Return ratios found in 3+ traditions
    sangam = sorted(
        [r for r, trads in ratio_traditions.items() if len(trads) >= 3],
        key=lambda r: tenney_height(r),
    )
    return sangam


# ── Precomputed Lookup ────────────────────────────────────────────────────

# Nearest simple ratio for cents values 0-1200 in 10-cent steps
# Format: {cents: (Fraction, tenney_height)}
NEAREST_HARMONIC: dict[int, tuple[Fraction, float]] = {}

def _build_lookup() -> None:
    """Build the nearest-harmonic lookup table."""
    for cents in range(0, 1201, 10):
        ratio = 2 ** (cents / 1200.0)
        best_frac = Fraction(1, 1)
        best_th = float('inf')

        for denom in range(1, 100):
            numer = round(ratio * denom)
            if numer < 1 or numer > 200:
                continue
            f = Fraction(numer, denom)
            actual_cents = 1200.0 * math.log2(float(f))
            if abs(actual_cents - cents) < 8:
                th = tenney_height(f)
                if th < best_th:
                    best_th = th
                    best_frac = f

        NEAREST_HARMONIC[cents] = (best_frac, best_th)

_build_lookup()
