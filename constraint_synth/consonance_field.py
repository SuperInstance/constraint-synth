"""
Consonance Field — The Landscape of Musical Beauty

The consonance field maps every point in Eisenstein lattice space to a consonance
value (how "resolved" or "beautiful" that interval sounds). This creates a 3D
landscape where:

- PEAKS = consonant intervals (perfect fifth, major third)
- VALLEYS = dissonant intervals (tritone, minor second)
- GRADIENT = direction toward more consonance (Spannung → resolution)
- SINGULARITIES = silence points (間, Ma) — gravity wells with no pitch

Agents and melodies NAVIGATE this landscape. Composing becomes moving through
a terrain of beauty.

Reference: Tenney (1983), Vos & van Vianen (1985), Sethares (2005)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional
import math

from .scales import (
    consonance_score, tenney_height, ratio_to_cents, cents_to_ratio,
    UNISON, PERFECT_FIFTH, Fraction,
)


@dataclass
class SilencePoint:
    """間 (Ma) — meaningful silence as a first-class musical element.
    
    Silence is not absence. It's presence-through-absence. In the consonance
    field, a silence point is a gravity well — it pulls neighboring notes
    toward stillness, toward breath.
    
    Attributes:
        position: Where in the melody/metric this silence occurs (ratio from phrase start)
        duration: How long the silence lasts (seconds)
        gravity: How strongly it pulls notes toward it (0-1)
        decay_mode: How the previous note's energy decays into the silence
            - "resonant": harmonics ring and fade naturally (弦, string)
            - "abrupt": sharp cutoff (打, percussive)
            - "breathing": tied to breath rhythm (息, vocal)
    """
    position: Fraction
    duration: float = 1.0
    gravity: float = 0.7
    decay_mode: str = "resonant"
    
    def tension_at(self, distance_cents: float) -> float:
        """How much tension this silence creates at a given distance.
        Closer notes feel more tension."""
        if distance_cents <= 0:
            return self.gravity
        return self.gravity * math.exp(-0.01 * distance_cents)


class ConsonanceField:
    """3D consonance landscape over the Eisenstein lattice.
    
    The field assigns a consonance value to every musical interval (expressed
    as a just-intonation ratio). This creates a continuous landscape where:
    
    - Height = consonance (how resolved/beautiful)
    - Steepness = tension (how much the music WANTS to move)
    - Singularities = silence points (間) with their own gravitational pull
    
    Usage:
        field = ConsonanceField()
        
        # How consonant is a perfect fifth?
        c = field.consonance_at(Fraction(3, 2))  # ≈ 0.275
        
        # Which direction leads to more consonance from the tritone?
        dx, dy = field.gradient_at(Fraction(45, 32))
        
        # Find shared consonance between two traditions
        shared = field.find_shared_consonance(scale_a, scale_b)
        
        # Add a silence point
        field.add_silence(SilencePoint(Fraction(1, 2), 2.0, 0.8, "breathing"))
    """
    
    def __init__(self, tradition: str = "universal"):
        self.tradition = tradition
        self.singularities: list[SilencePoint] = []
        self._harmonic_weight_cache: dict[Fraction, float] = {}
    
    def consonance_at(self, ratio: Fraction) -> float:
        """Consonance value at a given ratio. Higher = more consonant.
        
        Uses a hybrid metric:
        - Tenney height (harmonic simplicity) — primary
        - Euler gradus suavitatis — secondary correction
        - Singularity gravity — modifies value near silence points
        """
        base = consonance_score(ratio)
        
        # Modify near silence points
        if self.singularities:
            silence_bonus = 0.0
            for sp in self.singularities:
                # Silence affects nearby intervals by making them feel more "weighted"
                # (intervals near a silence point feel more deliberate, more meaningful)
                cents = abs(ratio_to_cents(ratio) - ratio_to_cents(sp.position))
                silence_bonus += sp.tension_at(cents) * 0.1
            base = min(1.0, base + silence_bonus)
        
        return base
    
    def dissonance_at(self, ratio: Fraction) -> float:
        """Inverse of consonance. Higher = more tense."""
        return 1.0 - self.consonance_at(ratio)
    
    def gradient_at(self, ratio: Fraction, step_cents: float = 100.0) -> tuple[float, float]:
        """Direction of increasing consonance in the pitch space.
        
        Returns (delta_up, delta_down) — how much consonance changes
        by moving up or down by step_cents. Positive values mean
        consonance increases in that direction.
        
        This IS Spannung (tension) — the music wants to move toward
        the direction of positive gradient.
        """
        current = self.consonance_at(ratio)
        
        # Step up - use cents_to_ratio for robust conversion
        cents_up = ratio_to_cents(ratio) + step_cents
        ratio_up = cents_to_ratio(cents_up)
        up = self.consonance_at(ratio_up) - current
        
        # Step down
        cents_down = ratio_to_cents(ratio) - step_cents
        if cents_down < 0:
            cents_down += 1200  # wrap around octave
        ratio_down = cents_to_ratio(cents_down)
        down = self.consonance_at(ratio_down) - current
        
        return (up, down)
    
    def find_nearest_peak(self, ratio: Fraction, search_range_cents: float = 200.0, resolution_cents: float = 5.0) -> Fraction:
        """Find the nearest consonance peak (interval of maximum beauty).
        
        Hill-climbs from the given ratio to find a local maximum in the
        consonance field. This is where the music wants to resolve TO.
        """
        from .scales import cents_to_ratio
        
        current_cents = ratio_to_cents(ratio)
        current_score = self.consonance_at(ratio)
        
        best_ratio = ratio
        best_score = current_score
        
        # Search in a range around the current position
        cents = current_cents - search_range_cents
        while cents < current_cents + search_range_cents:
            r = cents_to_ratio(cents)
            s = self.consonance_at(r)
            if s > best_score:
                best_score = s
                best_ratio = r
            cents += resolution_cents
        
        return best_ratio
    
    def tension_profile(self, intervals: list[Fraction]) -> list[float]:
        """Map a melody to a tension curve. Each interval gets a tension value.
        
        High tension = wants to resolve. Low tension = at rest.
        Useful for finding where 間 (silence points) should go —
        they belong at PEAK tension, where the need for breath is greatest.
        """
        return [self.dissonance_at(r) for r in intervals]
    
    def find_silence_positions(self, intervals: list[Fraction], threshold: float = 0.7) -> list[int]:
        """Find where silence points (間) should be placed in a melody.
        
        Returns indices of intervals where tension is above threshold.
        These are the moments where the music needs to BREATHE.
        """
        tensions = self.tension_profile(intervals)
        # Also consider tension peaks (local maxima)
        positions = []
        for i, t in enumerate(tensions):
            if t >= threshold:
                positions.append(i)
            # Local maximum
            elif i > 0 and i < len(tensions) - 1:
                if t > tensions[i-1] and t > tensions[i+1] and t > 0.5:
                    positions.append(i)
        return positions
    
    def add_silence(self, silence: SilencePoint):
        """Add a silence point (間) to the field."""
        self.singularities.append(silence)
    
    def find_shared_consonance(
        self, 
        scale_a_intervals: list[Fraction], 
        scale_b_intervals: list[Fraction],
        tolerance_cents: float = 50.0,
    ) -> list[tuple[Fraction, float]]:
        """Find intervals where two sets of intervals share high consonance.
        
        This is the mathematical basis of संगम (Sangam) — the confluence
        where two musical traditions discover they agree on beauty.
        
        Returns list of (ratio, consonance) tuples sorted by consonance.
        """
        shared = []
        
        for ra in scale_a_intervals:
            cents_a = ratio_to_cents(ra)
            for rb in scale_b_intervals:
                cents_b = ratio_to_cents(rb)
                if abs(cents_a - cents_b) <= tolerance_cents:
                    # Use the more consonant of the two
                    ca = self.consonance_at(ra)
                    cb = self.consonance_at(rb)
                    best = ra if ca >= cb else rb
                    score = max(ca, cb)
                    shared.append((best, score))
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for ratio, score in shared:
            key = float(ratio)
            if key not in seen:
                seen.add(key)
                unique.append((ratio, score))
        
        unique.sort(key=lambda x: x[1], reverse=True)
        return unique


# ─── Quick Demo ───

if __name__ == "__main__":
    from .scales import SCALES, consonance_overlap
    
    field = ConsonanceField()
    
    print("=== Consonance Field ===")
    print()
    
    # Key intervals
    intervals = {
        "Unison (1/1)": Fraction(1, 1),
        "Minor 3rd (6/5)": Fraction(6, 5),
        "Major 3rd (5/4)": Fraction(5, 4),
        "Perfect 4th (4/3)": Fraction(4, 3),
        "Tritone (45/32)": Fraction(45, 32),
        "Perfect 5th (3/2)": Fraction(3, 2),
        "Minor 7th (7/4)": Fraction(7, 4),
        "Octave (2/1)": Fraction(2, 1),
    }
    
    print("Consonance at key intervals:")
    for name, ratio in intervals.items():
        c = field.consonance_at(ratio)
        d = field.dissonance_at(ratio)
        bar = "█" * int(c * 40)
        print(f"  {name:>20s}: {c:.3f} consonance, {d:.3f} tension {bar}")
    
    print()
    print("Nearest consonance peak from tritone:")
    tritone = Fraction(45, 32)
    peak = field.find_nearest_peak(tritone)
    print(f"  {tritone} → {peak} (consonance: {field.consonance_at(peak):.3f})")
    
    print()
    print("Silence positions for a chromatic ascent:")
    chromatic = [Fraction(2) ** Fraction(i, 12) for i in range(1, 13)]
    positions = field.find_silence_positions(chromatic)
    print(f"  Tension peaks at indices: {positions}")
    print(f"  (These are where 間 belongs — moments of maximum tension needing breath)")
