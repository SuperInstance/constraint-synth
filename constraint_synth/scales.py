"""
Universal Scale System — Just Intonation Ratios + 12-TET Approximations

Every musical tradition's intervals stored as JUST INTONATION RATIOS (the lattice's
native language) alongside their 12-TET semitone approximations. This enables:

- Cross-tradition consonance comparison (Sangam — संगम)
- Lattice coordinate mapping for any interval
- Finding where two traditions AGREE on beauty

The key insight: semitones are a LOSSY COMPRESSION of just intonation.
Ratios are the lossless original. The lattice speaks in ratios.

References:
- Sagee (2021): 22 Śruti positions in Indian classical music
- Touma (1996): Maqam intervallic structures
- Malm (2000): Japanese traditional music intervals
- Tenney (1983): Consonance metrics via log-prime vectors
"""

from fractions import Fraction
from dataclasses import dataclass
from typing import Optional
import math


# ─── Fundamental Consonance Ratios ───
# These appear in nearly every musical tradition on Earth.
# Ordered by harmonic series position.

UNISON         = Fraction(1, 1)    # 1:1 — the root, 楽
OCTAVE         = Fraction(2, 1)    # 2:1 — universal
PERFECT_FIFTH  = Fraction(3, 2)    # 3:2 — universal
PERFECT_FOURTH = Fraction(4, 3)    # 4:3 — nearly universal (inverted fifth)
MAJOR_THIRD    = Fraction(5, 4)    # 5:4 — most traditions
MINOR_THIRD    = Fraction(6, 5)    # 6:5 — blues, raga, maqam, Japanese
MINOR_SEVENTH  = Fraction(7, 4)    # 7:4 — harmonic 7th, blues + gamaka territory
MAJOR_SECOND   = Fraction(9, 8)    # 9:8 — Pythagorean whole tone
MAJOR_SIXTH    = Fraction(5, 3)    # 5:3 — major scale sixth
MINOR_SIXTH    = Fraction(8, 5)    # 8:5 — minor scale sixth
MINOR_SEVENTH_FLAT = Fraction(9, 5)  # 9:5 — minor seventh (Just)
MAJOR_SEVENTH  = Fraction(15, 8)   # 15:8 — major scale leading tone

# ─── Tradition-Specific Ratios ───
# These exist in specific traditions and give them their unique color.

# Indian: 22 Śruti positions (from Bharata's Nāṭya Śāstra)
# Key komal (flattened) intervals
KOMAL_RE       = Fraction(16, 15)  # śruti position ~111.7 cents
KOMAL_GA_SMALL = Fraction(32, 27)  # ~294.1 cents — Bhairavi's characteristic komal ga
KOMAL_GA_LARGE = Fraction(6, 5)    # 316.1 cents — standard minor third
SHUDDHA_RE     = Fraction(9, 8)    # 203.9 cents — natural second
SHUDDHA_GA     = Fraction(5, 4)    # 386.3 cents — natural third
TIVRA_MA       = Fraction(45, 32)  # ~590.2 cents — sharp fourth (Yaman)
KOMAL_DHA      = Fraction(8, 5)    # 813.7 cents — Bhairavi's komal dha
KOMAL_NI       = Fraction(9, 5)    # 1017.6 cents — Bhairavi's komal ni
SHUDDHA_DHA    = Fraction(3, 2)    # reuse perfect fifth relative to pa
SHUDDHA_NI     = Fraction(5, 3)    # reuse major sixth relative to sa

# Arabic Maqam: Quarter-tone territory
# Between minor and major third
NEUTRAL_THIRD   = Fraction(11, 9)   # ~347.4 cents — quarter-tone between minor/major
RAST_THIRD      = Fraction(11, 9)   # Same — Maqam Rast's characteristic neutral third
HIJAZ_AUG_SEC   = Fraction(75, 64)  # ~275.4 cents — Hijaz's dramatic augmented second
BAYATI_SECOND   = Fraction(12, 11)  # ~150.6 cents — slightly flat second (Bayati)
SIKA_THIRD      = Fraction(11, 9)   # Neutral third (Sika)

# Japanese: Pentatonic with characteristic narrow intervals
HIRAJOSHI_MINOR_THIRD = Fraction(6, 5)   # 316 cents
HIRAJOSHI_TRITONE     = Fraction(45, 32) # ~590 cents — characteristic tension
IN_SCALE_THIRD   = Fraction(5, 4)         # major third in In scale
MIYAKO_FLAT_SEVEN = Fraction(7, 4)        # blue note in Miyako-bushi

# Blues: Blue notes as just-intonation ratios (NOT equal temperament)
BLUES_THIRD      = Fraction(6, 5)     # "blue third" — minor third bent toward major
BLUES_SEVENTH    = Fraction(7, 4)     # "blue seventh" — harmonic 7th, not minor 7th
BLUES_FLAT_FIFTH = Fraction(45, 32)   # tritone with just-intonation color

# Eastern European
HUNGARIAN_AUG_SEC = Fraction(75, 64)  # same as Hijaz augmented second!


# ─── Utility Functions ───

def ratio_to_cents(ratio: Fraction) -> float:
    """Convert a just-intonation ratio to cents (logarithmic measure)."""
    return 1200.0 * math.log2(float(ratio))


def cents_to_ratio(cents: float) -> Fraction:
    """Convert cents to nearest simple ratio (denominator ≤ 64)."""
    # Search for best simple fraction approximation
    value = 2 ** (cents / 1200.0)
    best = Fraction(1, 1)
    best_error = abs(ratio_to_cents(best) - cents)
    
    for d in range(1, 65):
        n = round(value * d)
        if n < 1:
            continue
        f = Fraction(n, d)
        err = abs(ratio_to_cents(f) - cents)
        if err < best_error:
            best = f
            best_error = err
    
    return best


def semitones_to_ratio(semitones: float) -> Fraction:
    """Convert 12-TET semitone count to nearest simple just-intonation ratio."""
    cents = semitones * 100.0
    return cents_to_ratio(cents)


def ratio_to_semitones(ratio: Fraction) -> float:
    """Convert a ratio to 12-TET semitone equivalent."""
    return 12.0 * math.log2(float(ratio))


def tenney_height(ratio: Fraction) -> float:
    """Tenney height: log₂(n) + log₂(d) for simplified fraction n/d.
    Lower = more consonant. Measures harmonic simplicity."""
    f = Fraction(ratio)  # ensure reduced
    # For ratios > 1, normalize to within one octave
    while f >= 2:
        f = f / 2
    while f < 1:
        f = f * 2
    return math.log2(f.numerator) + math.log2(f.denominator)


def euler_gradus_suavitatis(ratio: Fraction) -> float:
    """Euler's degree of sweetness: 1 + Σ(p-1) for all prime factors of n*d.
    Lower = more consonant. Complementary to Tenney height."""
    f = Fraction(ratio)
    product = f.numerator * f.denominator
    gs = 1.0
    # Factorize and sum (prime - 1)
    n = product
    p = 2
    while p * p <= n:
        while n % p == 0:
            gs += (p - 1)
            n //= p
        p += 1
    if n > 1:
        gs += (n - 1)
    return gs


def consonance_score(ratio: Fraction) -> float:
    """Combined consonance metric (0-1, higher = more consonant).
    Uses inverse Tenney height, normalized to [0, 1]."""
    th = tenney_height(ratio)
    # Tenney height ranges from 0 (unison) to ~10+ (very dissonant)
    # Map to [0, 1] with exponential decay
    return math.exp(-0.5 * th)


# ─── Scale Definitions ───

@dataclass(frozen=True)
class TraditionScale:
    """A musical scale defined by just-intonation ratios.
    
    Attributes:
        name: Human-readable name
        native_name: Name in original script/language
        tradition: Cultural tradition it belongs to
        intervals: List of ratios relative to the fundamental (1/1)
        description: What makes this scale unique
        characteristic_intervals: The intervals that define this scale's identity
    """
    name: str
    native_name: str
    tradition: str
    intervals: list[Fraction]
    description: str
    characteristic_intervals: list[Fraction]
    
    def at_degree(self, degree: int) -> Fraction:
        """Get the ratio at a specific scale degree (1-indexed)."""
        if degree == 1:
            return UNISON
        idx = (degree - 2) % len(self.intervals)
        octave_shift = (degree - 2) // len(self.intervals)
        return self.intervals[idx] * (2 ** octave_shift)
    
    def in_octave(self) -> list[Fraction]:
        """All intervals within one octave, sorted by ascending pitch."""
        # Normalize all to within [1, 2)
        result = []
        for r in self.intervals:
            norm = Fraction(r)
            while norm >= 2:
                norm = norm / 2
            while norm < 1:
                norm = norm * 2
            result.append(norm)
        result.sort(key=lambda r: float(r))
        return result
    
    def semitone_approximation(self) -> list[float]:
        """12-TET semitone approximation of this scale."""
        return [ratio_to_semitones(r) for r in self.intervals]
    
    def consonance_profile(self) -> list[float]:
        """Consonance score for each interval."""
        return [consonance_score(r) for r in self.intervals]
    
    def average_consonance(self) -> float:
        """Average consonance across all intervals."""
        scores = self.consonance_profile()
        return sum(scores) / len(scores) if scores else 0.0


# ─── The 27 Scales with Just Intonation Ratios ───

SCALES: dict[str, TraditionScale] = {
    # ─── Western ───
    "major": TraditionScale(
        name="Major",
        native_name="Dur / Ionian",
        tradition="western",
        intervals=[MAJOR_SECOND, MAJOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MAJOR_SIXTH, MAJOR_SEVENTH],
        description="The default scale of Western music. Bright, resolved.",
        characteristic_intervals=[MAJOR_THIRD, PERFECT_FIFTH],
    ),
    "natural_minor": TraditionScale(
        name="Natural Minor",
        native_name="Moll / Aeolian",
        tradition="western",
        intervals=[MAJOR_SECOND, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="The shadow of major. Contemplative, inward.",
        characteristic_intervals=[MINOR_THIRD, MINOR_SIXTH],
    ),
    "harmonic_minor": TraditionScale(
        name="Harmonic Minor",
        native_name="Harmonisches Moll",
        tradition="western",
        intervals=[MAJOR_SECOND, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MAJOR_SEVENTH],
        description="Minor with a raised 7th — creates an augmented 2nd (Spannung).",
        characteristic_intervals=[Fraction(75, 64)],  # the augmented 2nd between 6 and 7
    ),
    "melodic_minor": TraditionScale(
        name="Melodic Minor",
        native_name="Melodisches Moll",
        tradition="western",
        intervals=[MAJOR_SECOND, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MAJOR_SIXTH, MAJOR_SEVENTH],
        description="Minor ascending with raised 6th and 7th. Fluid identity.",
        characteristic_intervals=[MINOR_THIRD, MAJOR_SIXTH],
    ),
    "dorian": TraditionScale(
        name="Dorian",
        native_name="Dorisch",
        tradition="western",
        intervals=[MAJOR_SECOND, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MAJOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="Minor with a major 6th. Jazz workhorse. Neither bright nor dark.",
        characteristic_intervals=[MINOR_THIRD, MAJOR_SIXTH],
    ),
    "phrygian": TraditionScale(
        name="Phrygian",
        native_name="Phrygisch",
        tradition="western",
        intervals=[KOMAL_RE, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="Minor with a flat 2nd. Flamenco, dark tension.",
        characteristic_intervals=[KOMAL_RE],
    ),
    "lydian": TraditionScale(
        name="Lydian",
        native_name="Lydisch",
        tradition="western",
        intervals=[MAJOR_SECOND, MAJOR_THIRD, Fraction(45, 32), PERFECT_FIFTH, MAJOR_SIXTH, MAJOR_SEVENTH],
        description="Major with a sharp 4th. Floating, unresolved. Dreamlike.",
        characteristic_intervals=[Fraction(45, 32)],
    ),
    "mixolydian": TraditionScale(
        name="Mixolydian",
        native_name="Mixolydisch",
        tradition="western",
        intervals=[MAJOR_SECOND, MAJOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MAJOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="Major with a flat 7th. Blues-inflected. Rock staple.",
        characteristic_intervals=[MINOR_SEVENTH_FLAT],
    ),
    "locrian": TraditionScale(
        name="Locrian",
        native_name="Lokrisch",
        tradition="western",
        intervals=[KOMAL_RE, MINOR_THIRD, PERFECT_FOURTH, Fraction(45, 32), MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="The most unstable. Diminished fifth. Rarely resolves.",
        characteristic_intervals=[Fraction(45, 32), MINOR_THIRD],
    ),
    
    # ─── Indian Raga ───
    "bhairavi": TraditionScale(
        name="Bhairavi",
        native_name="भैरवी",
        tradition="indian",
        intervals=[KOMAL_RE, KOMAL_GA_SMALL, PERFECT_FOURTH, PERFECT_FIFTH, KOMAL_DHA, KOMAL_NI],
        description="Morning raga. All four komal (flat) notes. Devotional, dawn-light. The most compassionate raga.",
        characteristic_intervals=[KOMAL_GA_SMALL, KOMAL_DHA],
    ),
    "bhairav": TraditionScale(
        name="Bhairav",
        native_name="भैरव",
        tradition="indian",
        intervals=[KOMAL_RE, SHUDDHA_GA, PERFECT_FOURTH, PERFECT_FIFTH, KOMAL_DHA, SHUDDHA_NI],
        description="Serious, majestic morning raga. Komal re and dha frame a solemn grandeur.",
        characteristic_intervals=[KOMAL_RE, KOMAL_DHA],
    ),
    "yaman": TraditionScale(
        name="Yaman",
        native_name="यमन",
        tradition="indian",
        intervals=[SHUDDHA_RE, SHUDDHA_GA, TIVRA_MA, PERFECT_FIFTH, SHUDDHA_NI, MAJOR_SEVENTH],  # approximate with shuddha dha as 5th relative
        description="Evening raga. All shuddha (natural) with tivra ma (sharp 4th). Romantic, expansive.",
        characteristic_intervals=[TIVRA_MA],
    ),
    "kafi": TraditionScale(
        name="Kafi",
        native_name="काफी",
        tradition="indian",
        intervals=[SHUDDHA_RE, KOMAL_GA_LARGE, PERFECT_FOURTH, PERFECT_FIFTH, KOMAL_NI, MINOR_SEVENTH_FLAT],
        description="Light, playful raga. The komal ga gives it a folk-like sweetness.",
        characteristic_intervals=[KOMAL_GA_LARGE],
    ),
    "darbari": TraditionScale(
        name="Darbari",
        native_name="दरबारी",
        tradition="indian",
        intervals=[KOMAL_RE, KOMAL_GA_LARGE, PERFECT_FOURTH, PERFECT_FIFTH, KOMAL_DHA, KOMAL_NI],
        description="Court raga. Heavy, slow gamakas. Profound gravity. Built for royal audiences.",
        characteristic_intervals=[KOMAL_GA_LARGE, KOMAL_DHA],
    ),
    
    # ─── Arabic Maqam ───
    "hijaz": TraditionScale(
        name="Maqam Hijaz",
        native_name="مقام الحجاز",
        tradition="arabic",
        intervals=[MAJOR_SECOND, HIJAZ_AUG_SEC, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MAJOR_SEVENTH],
        description="The sound of the desert. Augmented 2nd between 2nd and 3rd. Dramatic, ancient.",
        characteristic_intervals=[HIJAZ_AUG_SEC],
    ),
    "bayati": TraditionScale(
        name="Maqam Bayati",
        native_name="مقام البياتي",
        tradition="arabic",
        intervals=[BAYATI_SECOND, MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="The most beloved maqam. Slightly flat 2nd. Warm, folk-like, nostalgic.",
        characteristic_intervals=[BAYATI_SECOND],
    ),
    "rast": TraditionScale(
        name="Maqam Rast",
        native_name="مقام الراست",
        tradition="arabic",
        intervals=[MAJOR_SECOND, RAST_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="The 'root' maqam. Neutral third — neither major nor minor. Proud, complete.",
        characteristic_intervals=[RAST_THIRD],
    ),
    
    # ─── Japanese ───
    "hirajoshi": TraditionScale(
        name="Hirajōshi",
        native_name="平調子",
        tradition="japanese",
        intervals=[MAJOR_SECOND, MINOR_THIRD, PERFECT_FIFTH, MINOR_SIXTH],
        description="Tense, ethereal pentatonic. The gap between 3rd and 5th creates characteristic 桜 (sakura) feeling.",
        characteristic_intervals=[MINOR_THIRD, MINOR_SIXTH],
    ),
    "in_scale": TraditionScale(
        name="In Scale",
        native_name="陰旋法",
        tradition="japanese",
        intervals=[MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SEVENTH_FLAT],
        description="Dark, reflective pentatonic. Used in shakuhachi meditation music.",
        characteristic_intervals=[MINOR_THIRD, MINOR_SEVENTH_FLAT],
    ),
    "miyako_bushi": TraditionScale(
        name="Miyako-bushi",
        native_name="都節",
        tradition="japanese",
        intervals=[MINOR_THIRD, PERFECT_FOURTH, MINOR_SIXTH, MINOR_SEVENTH_FLAT],
        description="Kabuki theater scale. The semitone between 1st and 2nd creates dramatic tension ( Spannung).",
        characteristic_intervals=[MINOR_THIRD, MINOR_SIXTH],
    ),
    
    # ─── Chinese ───
    "gong_mode": TraditionScale(
        name="Gong Mode",
        native_name="宮調式",
        tradition="chinese",
        intervals=[MAJOR_SECOND, MAJOR_THIRD, PERFECT_FIFTH, MAJOR_SIXTH],
        description="Five-note mode built on 宮 (gong, the emperor tone). Balanced, imperial.",
        characteristic_intervals=[MAJOR_THIRD, PERFECT_FIFTH],
    ),
    
    # ─── Blues ───
    "blues": TraditionScale(
        name="Blues",
        native_name="Blues",
        tradition="african_american",
        intervals=[MINOR_THIRD, PERFECT_FOURTH, BLUES_FLAT_FIFTH, PERFECT_FIFTH, BLUES_SEVENTH],
        description="Not a scale — a feeling. Blue notes bend toward consonance without arriving. The tritone as home.",
        characteristic_intervals=[BLUES_THIRD, BLUES_SEVENTH, BLUES_FLAT_FIFTH],
    ),
    
    # ─── Eastern European ───
    "hungarian_minor": TraditionScale(
        name="Hungarian Minor",
        native_name="Magyar moll",
        tradition="eastern_european",
        intervals=[MAJOR_SECOND, MINOR_THIRD, Fraction(45, 32), PERFECT_FIFTH, MINOR_SIXTH, MAJOR_SEVENTH],
        description="Minor with raised 4th AND major 7th. Two augmented 2nds. Wild, passionate.",
        characteristic_intervals=[Fraction(45, 32), MAJOR_SEVENTH],
    ),
    
    # ─── Pentatonic (Universal) ───
    "major_pentatonic": TraditionScale(
        name="Major Pentatonic",
        native_name="五声 (Wǔ shēng)",
        tradition="universal",
        intervals=[MAJOR_SECOND, MAJOR_THIRD, PERFECT_FIFTH, MAJOR_SIXTH],
        description="Found in Chinese, Celtic, African, Appalachian, and children's songs worldwide. No dissonance possible.",
        characteristic_intervals=[MAJOR_THIRD, PERFECT_FIFTH],
    ),
    "minor_pentatonic": TraditionScale(
        name="Minor Pentatonic",
        native_name="小五声",
        tradition="universal",
        intervals=[MINOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH, MINOR_SEVENTH_FLAT],
        description="Rock and blues foundation. Every note sounds good together. Forgiving, warm.",
        characteristic_intervals=[MINOR_THIRD, MINOR_SEVENTH_FLAT],
    ),
    
    # ─── Whole Tone (Ambient/Impressionist) ───
    "whole_tone": TraditionScale(
        name="Whole Tone",
        native_name="Ton entier",
        tradition="western_impressionist",
        intervals=[MAJOR_SECOND, Fraction(5, 4) * Fraction(9, 8), Fraction(45, 32), Fraction(3, 2) * Fraction(9, 8), MAJOR_SEVENTH * Fraction(9, 8)],
        description="No leading tone, no resolution. Debussy's dream. Every step is equal. Floating.",
        characteristic_intervals=[],  # no characteristic interval — that's the point
    ),
}


# ─── Cross-Tradition Analysis ───

def consonance_overlap(scale_a: str, scale_b: str, tolerance_cents: float = 50.0) -> list[tuple[Fraction, str, str]]:
    """Find intervals where two traditions share near-consonance.
    
    Returns list of (shared_ratio, degree_in_a, degree_in_b) tuples.
    This is संगम (Sangam) — confluence of traditions.
    
    Parameters
    ----------
    scale_a, scale_b : str
        Scale names from SCALES dict
    tolerance_cents : float
        Maximum cents deviation to consider "the same" interval (default: 50 cents)
    """
    a = SCALES[scale_a]
    b = SCALES[scale_b]
    
    shared = []
    for i, ra in enumerate(a.intervals):
        cents_a = ratio_to_cents(ra)
        for j, rb in enumerate(b.intervals):
            cents_b = ratio_to_cents(rb)
            deviation = abs(cents_a - cents_b)
            if deviation <= tolerance_cents:
                # Use the simpler ratio (lower Tenney height)
                simpler = ra if tenney_height(ra) < tenney_height(rb) else rb
                shared.append((simpler, a.name, b.name))
    
    return shared


def find_sangam(traditions: list[str], tolerance_cents: float = 50.0) -> list[Fraction]:
    """Find intervals shared by ALL given traditions.
    
    This is the deep consonance — where every tradition agrees.
    These intervals become the convergence points (Σημείο Σύγκλισης)
    for multi-agent creative sessions.
    """
    if len(traditions) < 2:
        return []
    
    # Start with all intervals from first tradition
    candidates = set()
    for ratio in SCALES[traditions[0]].intervals:
        candidates.add(ratio)
    
    # Intersect with each subsequent tradition
    for trad in traditions[1:]:
        surviving = set()
        trad_cents = [ratio_to_cents(r) for r in SCALES[trad].intervals]
        
        for candidate in candidates:
            c_cents = ratio_to_cents(candidate)
            for t_cents in trad_cents:
                if abs(c_cents - t_cents) <= tolerance_cents:
                    surviving.add(candidate)
                    break
        
        candidates = surviving
    
    return sorted(candidates, key=lambda r: tenney_height(r))


def tradition_distance(scale_a: str, scale_b: str) -> float:
    """How far apart are two traditions in consonance space?
    Lower = more shared intervals = easier to collaborate.
    
    Returns average cents deviation of nearest-match intervals.
    """
    a = SCALES[scale_a]
    b = SCALES[scale_b]
    
    total_deviation = 0.0
    comparisons = 0
    
    for ra in a.intervals:
        cents_a = ratio_to_cents(ra)
        min_dev = min(abs(cents_a - ratio_to_cents(rb)) for rb in b.intervals)
        total_deviation += min_dev
        comparisons += 1
    
    return total_deviation / comparisons if comparisons > 0 else float('inf')


# ─── Export ───

def list_traditions() -> dict[str, list[str]]:
    """Group scales by cultural tradition."""
    traditions: dict[str, list[str]] = {}
    for key, scale in SCALES.items():
        traditions.setdefault(scale.tradition, []).append(key)
    return traditions


def get_scale(name: str) -> Optional[TraditionScale]:
    """Get a scale by name (case-insensitive)."""
    return SCALES.get(name.lower())


if __name__ == "__main__":
    # Demo: Find where Bhairavi and Hijaz agree
    print("=" * 60)
    print("संगम (Sangam) — Cross-Tradition Consonance")
    print("=" * 60)
    
    # Bhairavi × Hijaz
    shared = consonance_overlap("bhairavi", "hijaz")
    print(f"\nBhairavi × Hijaz shared intervals:")
    for ratio, a, b in shared:
        print(f"  {float(ratio):.4f} ({ratio_to_cents(ratio):.1f}¢) — consonance: {consonance_score(ratio):.3f}")
    
    # Multi-tradition Sangam
    universal = find_sangam(["major", "bhairavi", "hijaz", "hirajoshi", "blues"])
    print(f"\nUniversal Sangam (Major ∩ Bhairavi ∩ Hijaz ∩ Hirajoshi ∩ Blues):")
    for ratio in universal:
        print(f"  {float(ratio):.4f} ({ratio_to_cents(ratio):.1f}¢) — {consonance_score(ratio):.3f}")
    
    # Tradition distance matrix
    print(f"\nTradition Distance (avg cents deviation):")
    key_scales = ["major", "bhairavi", "hijaz", "hirajoshi", "blues", "dorian"]
    print(f"  {'':>12s}", end="")
    for k in key_scales:
        print(f"  {k[:10]:>10s}", end="")
    print()
    for ka in key_scales:
        print(f"  {ka:>12s}", end="")
        for kb in key_scales:
            d = tradition_distance(ka, kb)
            print(f"  {d:>10.1f}", end="")
        print()
    
    # Consonance ranking
    print(f"\nConsonance ranking (most consonant → least):")
    ranked = sorted(SCALES.items(), key=lambda x: x[1].average_consonance(), reverse=True)
    for name, scale in ranked[:10]:
        print(f"  {name:>20s} ({scale.native_name:>12s}): {scale.average_consonance():.3f} — {scale.tradition}")
