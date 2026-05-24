"""
Rhythmic Consonance — The 3/2 Principle in Time

If 3:2 is the most consonant non-identity in pitch (perfect fifth),
then 3-in-2 is the most consonant non-identity in rhythm (hemiola).

This module extends the consonance field into the time domain, applying
the SAME mathematical framework to rhythm that we apply to harmony.

Key insight (Cowell 1930, Nancarrow 1947): the overtone series generates
both pitch AND rhythm. Frequency ratios below ~16Hz are perceived as
rhythm; above ~16Hz as pitch. The same ratios apply in both domains.

References:
- Cowell (1930): New Musical Resources
- Toussaint (2005): Euclidean Algorithm Generates Traditional Rhythms
- Nancarrow: Study No. 37 (tempo ratios = just-intonation pitch ratios)
- Arom (1991): African Polyphony and Polyrhythm
"""

from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional
import math

from .scales import consonance_score, tenney_height


# ─── Core Rhythmic Ratios ───

# The same ratios that define pitch consonance define rhythmic consonance
# Below ~16Hz, the brain perceives frequency as rhythm, not pitch

DUPLE = Fraction(2, 1)       # The heartbeat. March. Basic pulse.
TRIPLE = Fraction(3, 1)      # The waltz. Sacred meter. Tempus perfectum.
HEMIOLA = Fraction(3, 2)     # 3 in 2. The PERFECT FIFTH OF RHYTHM.
SESQUIALTERA = Fraction(3, 2)  # Same ratio, Renaissance term
QUINTUPLET = Fraction(5, 4)   # Major third in rhythm — challenging but coherent
SEPTUPLET = Fraction(7, 4)    # Minor seventh in rhythm — complex, jazzy


def rhythmic_consonance(ratio: Fraction) -> float:
    """How 'natural' does a rhythmic ratio feel?
    
    Uses the SAME consonance function as pitch — because the brain
    processes periodicity the same way whether it's 440Hz (pitch)
    or 2Hz (rhythm).
    
    3:2 (hemiola) has the highest non-trivial consonance.
    2:1 (duple) is identity (octave equivalent — trivially consonant).
    7:4 (septuplet) is complex, like a minor seventh.
    """
    return consonance_score(ratio)


# ─── Aksak (Asymmetric Meter) Generation ───

def generate_aksak(total_beats: int) -> list[int]:
    """Generate an aksak (2s and 3s) grouping for any meter length.
    
    Uses the Euclidean algorithm (Björklund 2003, Toussaint 2005)
    to distribute beats as evenly as possible.
    
    The principle: distribute 'threes' among 'twos' as evenly as possible.
    This creates the characteristic "limping" feel of Balkan aksak rhythms.
    
    Examples:
        5 → [3, 2]           — limping waltz
        7 → [2, 2, 3]        — the classic aksak (Rūpaka tāla)
        9 → [2, 2, 2, 3]     — extended aksak
        11 → [2, 2, 3, 2, 2] — Macedonian folk pattern
        13 → [3, 2, 3, 2, 3] — progressive rock territory
    
    Parameters
    ----------
    total_beats : int
        Total number of beats in the asymmetric meter
    
    Returns
    -------
    list[int]
        Grouping pattern (only 2s and 3s)
    """
    if total_beats <= 0:
        return []
    if total_beats <= 2:
        return [total_beats]
    if total_beats == 3:
        return [3]
    
    # Count how many 3s we need
    num_threes = total_beats // 3
    remainder = total_beats % 3
    
    if remainder == 0 and num_threes > 1:
        # All threes — no asymmetry (e.g., 9 = 3+3+3)
        # Consider mixing with 2s for more interest
        # Use Euclidean distribution
        return _euclidean_rhythm(total_beats, num_threes)
    
    if remainder == 1 and num_threes >= 1:
        # Convert one 3+1 into 2+2
        # e.g., 7 = 3+3+1 → use 2+2+3 instead
        num_threes -= 1
        num_twos = (total_beats - num_threes * 3) // 2
    else:
        num_twos = remainder // 2 if remainder > 0 else 0
    
    # Distribute threes among twos using Euclidean principle
    return _distribute_groups(num_twos, num_threes)


def _euclidean_rhythm(total: int, pulses: int) -> list[int]:
    """Euclidean rhythm: distribute 'pulses' beats evenly across 'total' beats.
    Returns grouping as list of 2s and 3s."""
    if pulses == 0:
        return [2] * (total // 2) if total % 2 == 0 else [2] * (total // 2) + [2]
    
    # Use Björklund's algorithm
    pattern = [1] * pulses + [0] * (total - pulses)
    
    # The Euclidean algorithm
    while True:
        zeros = [i for i, x in enumerate(pattern) if x == 0]
        ones = [i for i, x in enumerate(pattern) if x == 1]
        
        if len(zeros) <= 1 or len(ones) <= 1:
            break
        
        min_len = min(len(ones), len(zeros))
        
        # Pair them
        new_pattern = []
        for i in range(min_len):
            new_pattern.append([pattern[ones[i]], pattern[zeros[i]]])
        
        # Add remainder
        remainder = pattern[min_len * 2:]
        if remainder:
            new_pattern.append(remainder)
        
        # Flatten one level
        flat = []
        for item in new_pattern:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        
        if flat == pattern:
            break
        pattern = flat
    
    # Convert pattern to groupings
    groups = []
    count = 0
    for beat in pattern:
        count += 1
        if beat == 1 and count > 0:
            if count > 1:
                groups.append(count)
            count = 0
    if count > 0:
        groups.append(count + 1)  # include the last pulse
    
    # Validate: all groups should be 2 or 3
    # If any group is > 3, split it
    result = []
    for g in groups:
        while g > 3:
            result.append(3)
            g -= 3
        if g > 0:
            result.append(g)
    
    return result if result else [total]


def _distribute_groups(twos: int, threes: int) -> list[int]:
    """Distribute threes evenly among twos."""
    total = twos + threes
    if total == 0:
        return []
    
    # Use Euclidean distribution
    result = [2] * twos + [3] * threes
    
    # Interleave for maximum evenness
    if threes == 0:
        return [2] * twos
    if twos == 0:
        return [3] * threes
    
    # Find the spacing
    output = []
    pos_threes = 0
    pos_twos = 0
    
    for i in range(total):
        # Place a 3 if it's time
        t_pos = pos_threes * twos // threes if threes > 0 else total
        if pos_threes < threes and i >= t_pos:
            output.append(3)
            pos_threes += 1
        elif pos_twos < twos:
            output.append(2)
            pos_twos += 1
        else:
            output.append(3)
            pos_threes += 1
    
    return output


# ─── World Rhythm Patterns ───

@dataclass(frozen=True)
class RhythmicTradition:
    """A rhythmic pattern from a specific cultural tradition.
    
    Attributes:
        name: Common name
        native_name: Name in original language
        tradition: Cultural origin
        pattern: Beat grouping (e.g., [3, 2, 2] for rūpaka tāla)
        description: What this rhythm expresses
        consonance: How consonant the rhythmic ratio feels
    """
    name: str
    native_name: str
    tradition: str
    pattern: list[int]
    description: str
    
    @property
    def total_beats(self) -> int:
        return sum(self.pattern)
    
    @property
    def consonance(self) -> float:
        """Average consonance of the rhythmic grouping."""
        scores = []
        for i, group in enumerate(self.pattern):
            # Compare each group to its neighbors
            for j in range(i + 1, len(self.pattern)):
                ratio = Fraction(max(group, self.pattern[j]), min(group, self.pattern[j]))
                scores.append(consonance_score(ratio))
        return sum(scores) / len(scores) if scores else 1.0
    
    @property
    def density(self) -> float:
        """Proportion of 3-groups (higher = more asymmetric)."""
        threes = sum(1 for g in self.pattern if g == 3)
        return threes / len(self.pattern) if self.pattern else 0.0


RHYTHMS: dict[str, RhythmicTradition] = {
    # ─── African ───
    "agbekor": RhythmicTradition(
        name="Agbekor Bell",
        native_name="Agbekor",
        tradition="ewe",
        pattern=[3, 3, 2],
        description="Ewe war drum rhythm. 3+3+2 = 8. The 3s fight the underlying 4/4.",
    ),
    "kpanlogo": RhythmicTradition(
        name="Kpanlogo",
        native_name="Kpanlogo",
        tradition="ga",
        pattern=[3, 3, 4],
        description="Ga people (Ghana). Recreational dance. The 4 anchors the 3s.",
    ),
    
    # ─── Indian ───
    "rupaka": RhythmicTradition(
        name="Rūpaka Tāla",
        native_name="रूपक ताल",
        tradition="hindustani",
        pattern=[3, 2, 2],
        description="7 beats: drutam + drutam + laghu. The 3 leads — resolved by two 2s.",
    ),
    "jhaptal": RhythmicTradition(
        name="Jhaptāla",
        native_name="झपताल",
        tradition="hindustani",
        pattern=[2, 3, 2, 3],
        description="10 beats: symmetric interlocking of 2s and 3s.",
    ),
    "adi": RhythmicTradition(
        name="Ādi Tāla",
        native_name="आदि ताल",
        tradition="carnatic",
        pattern=[4, 2, 2],  # technically laghu(4) + drutam(2) + drutam(2)
        description="8 beats. The 'father' of Carnatic tālas. Internal 4 = 2+2.",
    ),
    "misra_chapu": RhythmicTradition(
        name="Miśra Chāpu",
        native_name="மிஸ்ர சாப்பு",
        tradition="carnatic",
        pattern=[3, 2, 2],
        description="7 beats. Carnatic counterpart to rūpaka. Flowing, dance-like.",
    ),
    
    # ─── Balkan / Turkish ───
    "paidushko": RhythmicTradition(
        name="Paidushko",
        native_name="Пайдушко",
        tradition="bulgarian",
        pattern=[2, 3],
        description="5 beats. Macedonian/Bulgarian folk dance. Quick-limp.",
    ),
    "lesnoto": RhythmicTradition(
        name="Lesnoto",
        native_name="Лесното",
        tradition="macedonian",
        pattern=[3, 2, 2],
        description="7 beats. 'The easy one.' But it's not easy for outsiders.",
    ),
    "devetorka": RhythmicTradition(
        name="Devetorka",
        native_name="Деветорка",
        tradition="macedonian",
        pattern=[2, 2, 2, 3],
        description="9 beats. The 3 at the end is the aksak twist.",
    ),
    "dajchovo": RhythmicTradition(
        name="Dajčovo",
        native_name="Дайчово",
        tradition="bulgarian",
        pattern=[2, 2, 3, 2],
        description="9 beats with the 3 in the middle. 'Pounding' dance.",
    ),
    "_eleven": RhythmicTradition(
        name="11-beat Aksak",
        native_name="Aksak",
        tradition="turkish",
        pattern=[3, 3, 3, 2],
        description="11 beats. Three 3s and a 2. The 'limp' at the end is intense.",
    ),
    
    # ─── Western Art Music ───
    "hemiola": RhythmicTradition(
        name="Hemiola",
        native_name="Ἡμιόλιος",
        tradition="western",
        pattern=[3, 3],
        description="3+3 in the space of 2+2+2. The Greeks named it for the 5th. Renaissance composers weaponized it.",
    ),
    "courante": RhythmicTradition(
        name="Courante Hemiola",
        native_name="Courante",
        tradition="baroque",
        pattern=[3, 2],
        description="5 beats of hemiola feel in 3/2 meter. Bach's courantes live here.",
    ),
    "siciliana": RhythmicTradition(
        name="Siciliana",
        native_name="Siciliana",
        tradition="baroque",
        pattern=[3, 2, 3, 2],
        description="12/8 meter felt as 3+2+3+2. Pastoral, swaying. The 3s and 2s alternate.",
    ),
    
    # ─── Modern / Experimental ───
    "nancarrow_3_2": RhythmicTradition(
        name="Nancarrow 3:2 Tempo",
        native_name="Study No. 37",
        tradition="modernist",
        pattern=[3, 2],
        description="Two voices at tempo ratio 3:2 — horizontal perfect fifth. Nancarrow's insight.",
    ),
}


# ─── Nancarrow Canon Generator ───

@dataclass
class NancarrowVoice:
    """A single voice in a polytemporal canon."""
    tempo_ratio: Fraction    # Relative to fundamental (1/1)
    pitch: int = 60          # MIDI note
    velocity: int = 80       # Volume
    pattern: list[int] = field(default_factory=lambda: [1])  # rhythmic pattern


class NancarrowCanon:
    """Polytemporal canon where tempo ratios = just-intonation pitch ratios.
    
    Based on Nancarrow's Study No. 37: each voice moves at a tempo
    corresponding to a just-intonation interval. The perfect fifth voice
    moves at 3/2 the speed of the fundamental voice.
    
    This makes the vertical 3/2 (harmony) and horizontal 3/2 (rhythm)
    literally the same mathematical relationship.
    """
    
    def __init__(self, voices: list[NancarrowVoice]):
        self.voices = voices
    
    @classmethod
    def from_chord(cls, ratios: list[Fraction], base_pitch: int = 60) -> "NancarrowCanon":
        """Create a canon from a chord of just-intonation ratios.
        
        Each ratio becomes both a pitch interval AND a tempo ratio.
        """
        voices = []
        for i, ratio in enumerate(ratios):
            # Convert ratio to MIDI pitch (approximate)
            from .scales import ratio_to_semitones
            semitones = ratio_to_semitones(ratio)
            pitch = base_pitch + round(semitones)
            
            voices.append(NancarrowVoice(
                tempo_ratio=ratio,
                pitch=pitch,
                velocity=max(40, 100 - i * 10),
            ))
        return cls(voices)
    
    def render(self, duration_beats: float, sr: int = 44100) -> dict[int, list[tuple[float, float]]]:
        """Render all voices. Returns {voice_index: [(start_time, duration)]}.
        
        Each voice plays at its own tempo. Voice with tempo_ratio 3/2
        plays 50% faster than the fundamental voice.
        """
        result = {}
        fundamental_bpm = 120  # base tempo
        
        for i, voice in enumerate(self.voices):
            voice_bpm = fundamental_bpm * float(voice.tempo_ratio)
            beat_duration = 60.0 / voice_bpm  # seconds per beat
            
            events = []
            time = 0.0
            beat_in_pattern = 0
            
            while time < duration_beats * 60.0 / fundamental_bpm:
                pattern_beat = voice.pattern[beat_in_pattern % len(voice.pattern)]
                event_duration = pattern_beat * beat_duration
                events.append((time, event_duration))
                time += event_duration
                beat_in_pattern += 1
            
            result[i] = events
        
        return result


# ─── Tension in Rhythm ───

def rhythmic_tension(pattern: list[int]) -> list[float]:
    """Compute tension at each beat in a rhythmic pattern.
    
    Tension is highest where the pattern deviates most from pure duple.
    A beat of 3 in a field of 2s creates maximum tension.
    """
    total = sum(pattern)
    avg = total / len(pattern) if pattern else 2.0
    
    tensions = []
    for group in pattern:
        # Tension = deviation from duple
        deviation = abs(group - 2)
        tensions.append(deviation / 2.0)  # normalize to [0, 1]
    
    return tensions


def find_syncopation_points(pattern: list[int]) -> list[int]:
    """Find beats where rhythmic tension peaks — good for syncopation.
    
    In aksak rhythms, the '3' group creates a natural accent.
    Syncopation works best AGAINST this accent — on the off-beats of the 3.
    """
    tensions = rhythmic_tension(pattern)
    if not tensions:
        return []
    
    max_tension = max(tensions)
    return [i for i, t in enumerate(tensions) if t == max_tension]


# ─── Export ───

def list_rhythmic_traditions() -> dict[str, list[str]]:
    """Group rhythms by cultural tradition."""
    traditions: dict[str, list[str]] = {}
    for key, rhythm in RHYTHMS.items():
        traditions.setdefault(rhythm.tradition, []).append(key)
    return traditions


if __name__ == "__main__":
    print("=== Aksak Generation ===")
    for n in [5, 7, 8, 9, 11, 13]:
        pattern = generate_aksak(n)
        print(f"  {n:2d} beats → {'+'.join(str(g) for g in pattern)}")
    
    print("\n=== World Rhythms ===")
    for name, rhythm in RHYTHMS.items():
        pattern_str = '+'.join(str(g) for g in rhythm.pattern)
        tension = rhythmic_tension(rhythm.pattern)
        print(f"  {name:>16s} ({rhythm.native_name:>12s}): {pattern_str:>10s} tension={[f'{t:.1f}' for t in tension]} consonance={rhythm.consonance:.3f}")
    
    print("\n=== Nancarrow Canon (Just-Intonation Tempos) ===")
    from fractions import Fraction as F
    canon = NancarrowCanon.from_chord([
        F(1, 1), F(5, 4), F(3, 2), F(2, 1),  # fundamental, major 3rd, fifth, octave
    ])
    for i, voice in enumerate(canon.voices):
        print(f"  Voice {i}: ratio {voice.tempo_ratio}, pitch {voice.pitch}")
    
    print("\n=== Rhythmic Consonance of Key Ratios ===")
    ratios = [
        ("2:1 (duple/octave)", F(2, 1)),
        ("3:2 (hemiola/fifth)", F(3, 2)),
        ("4:3 (cross-rhythm/fourth)", F(4, 3)),
        ("5:4 (quintuplet/3rd)", F(5, 4)),
        ("5:3 (复合)", F(5, 3)),
        ("7:4 (septuplet/min7)", F(7, 4)),
    ]
    for name, ratio in ratios:
        c = rhythmic_consonance(ratio)
        bar = "█" * int(c * 40)
        print(f"  {name:>30s}: {c:.3f} {bar}")
