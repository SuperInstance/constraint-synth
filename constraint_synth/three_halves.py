"""
Three Halves — The 3:2 Isomorphism Between Pitch and Rhythm

"If 3:2 is the most consonant non-identity in pitch (perfect fifth),
then 3-in-2 is the most consonant non-identity in rhythm (hemiola)."

This module implements three deep musical concepts:

1. ThreeHalves: Unified conversion between melodic and rhythmic phrases
   using the SAME ratios interpreted differently (pitch vs time).

2. MeantoneSimulator: Quarter-comma meantone tuning demonstration.
   Hear why D major sounded triumphant before equal temperament.
   See which keys have wolf intervals.

3. NancarrowStudy37: 12-voice canon with just-intonation tempo ratios.
   Each voice plays the same melody at its characteristic tempo.
   Alignment points create "temporal consonance" — the moment when
   the rhythm resolves, analogous to harmonic resolution.

Key insight (Cowell 1930, Nancarrow 1947): the overtone series generates
both pitch AND rhythm. Frequency ratios below ~16Hz are perceived as rhythm;
above ~16Hz as pitch. The SAME ratios apply in both domains.

References:
- Cowell (1930): New Musical Resources
- Nancarrow: Study No. 37 (tempo ratios = just-intonation pitch ratios)
- THREE-HALVES.md: Deep research on the 3/2 principle across traditions
"""

from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
import math
import numpy as np
from enum import Enum


# ─── Import from existing modules ───

from constraint_synth.scales import (
    consonance_score, ratio_to_cents, cents_to_ratio,
    ratio_to_semitones, PERFECT_FIFTH, MAJOR_THIRD,
    PERFECT_FOURTH, OCTAVE, MINOR_THIRD, MAJOR_SIXTH,
    MINOR_SIXTH, MAJOR_SECOND, MAJOR_SEVENTH, MINOR_SEVENTH_FLAT,
)


# ─── Three Halves: Pitch-Rhythm Isomorphism ───

@dataclass(frozen=True)
class MelodicNote:
    """A note in a melodic phrase."""
    ratio: Fraction          # Just-intonation ratio relative to fundamental
    duration: Fraction = Fraction(1, 1)  # Duration as a fraction of a beat

    @property
    def midi_pitch(self) -> int:
        """Approximate MIDI pitch (within one octave of 60)."""
        semitones = ratio_to_semitones(self.ratio)
        return 60 + round(semitones)

    def frequency(self) -> float:
        """Frequency in Hz (with A4=440 as fundamental for ratio=1/1)."""
        return 440.0 * float(self.ratio)

    def hz(self, fundamental_hz: float = 440.0) -> float:
        """Frequency in Hz for a given fundamental."""
        return fundamental_hz * float(self.ratio)


@dataclass
class RhythmicEvent:
    """A rhythmic event derived from a melodic ratio."""
    duration: Fraction      # Duration as fraction of a beat (SAME as pitch ratio!)
    velocity: int = 80      # MIDI velocity


@dataclass
class UnifiedPhrase:
    """A musical phrase that exists simultaneously as pitch and rhythm."""
    notes: list[MelodicNote]
    tempo_bpm: float = 120.0

    @property
    def fundamental(self) -> Fraction:
        """The ratio of the first note (anchor)."""
        return self.notes[0].ratio if self.notes else Fraction(1, 1)

    @property
    def ratios(self) -> list[Fraction]:
        """All pitch ratios in the phrase."""
        return [n.ratio for n in self.notes]

    @property
    def durations(self) -> list[Fraction]:
        """All durations in the phrase."""
        return [n.duration for n in self.notes]

    @property
    def total_beats(self) -> Fraction:
        """Total length in beats."""
        return sum(self.durations)

    def to_rhythmic_phrase(self) -> list[RhythmicEvent]:
        """Convert to rhythmic phrase using the SAME ratios as durations.
        This is the KEY isomorphism: pitch ratio = rhythmic ratio.
        """
        events = []
        for note in self.notes:
            # Use the pitch ratio as the time ratio!
            # 3:2 (perfect fifth) becomes 3 beats in the space of 2.
            events.append(RhythmicEvent(duration=note.ratio, velocity=80))
        return events

    def to_melodic_phrase(self) -> list[tuple[int, Fraction]]:
        """Extract as (midi_pitch, duration) tuples."""
        return [(n.midi_pitch, n.duration) for n in self.notes]


class ThreeHalves:
    """Demonstrates the isomorphism between pitch 3/2 and rhythm 3/2.

    The core insight: the SAME mathematical ratio (3:2) produces:
    - Vertically: the perfect fifth — the most consonant non-identity interval
    - Horizontally: hemiola — the most groovy non-identity rhythm

    This class lets you:
    1. Convert melodic phrases to rhythmic phrases using the same ratios
    2. Convert rhythmic phrases to melodic phrases
    3. Render both simultaneously to HEAR the isomorphism
    """

    def __init__(self, fundamental_hz: float = 440.0):
        self.fundamental_hz = fundamental_hz

    def melody_to_rhythm(self, melody: list[MelodicNote]) -> list[RhythmicEvent]:
        """Convert a melodic phrase to a rhythmic phrase.

        The pitch ratios become time ratios. A perfect fifth (3:2) becomes
        3 beats in the space of 2 beats — hemiola!

        Args:
            melody: List of melodic notes with pitch ratios

        Returns:
            List of rhythmic events where durations = pitch ratios
        """
        phrase = UnifiedPhrase(notes=melody)
        return phrase.to_rhythmic_phrase()

    def rhythm_to_melody(self, rhythm: list[RhythmicEvent]) -> list[MelodicNote]:
        """Convert a rhythmic phrase to a melodic phrase.

        The time ratios become pitch ratios. A hemiola (3 beats in 2) becomes
        a perfect fifth (3:2)!

        Args:
            rhythm: List of rhythmic events with duration ratios

        Returns:
            List of melodic notes where pitch ratios = durations
        """
        melody = []
        for event in rhythm:
            # Use the duration ratio as the pitch ratio!
            melody.append(MelodicNote(ratio=event.duration, duration=Fraction(1, 1)))
        return melody

    def render_isomorphism(
        self,
        melody: list[MelodicNote],
        sr: int = 44100,
        tempo_bpm: float = 120.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render both the melodic AND rhythmic interpretation.

        Returns (audio_pitch, audio_rhythm) as numpy arrays.
        You can then play both simultaneously to hear the isomorphism.

        Args:
            melody: The melodic phrase to render in both domains
            sr: Sample rate
            tempo_bpm: Tempo in beats per minute

        Returns:
            Tuple of (pitch_audio, rhythm_audio) arrays
        """
        # Render melodic version (pitch domain)
        pitch_audio = self._render_melody(melody, sr, tempo_bpm)

        # Render rhythmic version (time domain)
        rhythm = self.melody_to_rhythm(melody)
        rhythm_audio = self._render_rhythm(rhythm, sr, tempo_bpm)

        return pitch_audio, rhythm_audio

    def _render_melody(
        self,
        melody: list[MelodicNote],
        sr: int = 44100,
        tempo_bpm: float = 120.0,
    ) -> np.ndarray:
        """Render a melodic phrase as audio."""
        beat_duration = 60.0 / tempo_bpm  # seconds per beat

        # Calculate total duration
        total_beats = float(sum(note.duration for note in melody))
        total_samples = int(total_beats * beat_duration * sr)

        audio = np.zeros(total_samples)

        sample_pos = 0
        for note in melody:
            duration_sec = float(note.duration) * beat_duration
            duration_samples = int(duration_sec * sr)

            if duration_samples <= 0:
                continue

            # Generate sine wave at this pitch
            t = np.linspace(0, duration_sec, duration_samples, endpoint=False)
            freq = note.hz(261.63)  # C4 as fundamental
            wave = 0.5 * np.sin(2 * np.pi * freq * t)

            # Apply envelope to avoid clicks
            envelope = np.ones_like(wave)
            fade_len = min(100, duration_samples // 10)
            if fade_len > 0:
                envelope[:fade_len] = np.linspace(0, 1, fade_len)
                envelope[-fade_len:] = np.linspace(1, 0, fade_len)

            wave = wave * envelope

            # Place in audio
            end_pos = min(sample_pos + duration_samples, total_samples)
            actual_len = end_pos - sample_pos
            if actual_len > 0:
                audio[sample_pos:end_pos] += wave[:actual_len]

            sample_pos += duration_samples

        return audio

    def _render_rhythm(
        self,
        rhythm: list[RhythmicEvent],
        sr: int = 44100,
        tempo_bpm: float = 120.0,
        click_freq: float = 1000.0,
    ) -> np.ndarray:
        """Render a rhythmic phrase as clicks (time domain)."""
        beat_duration = 60.0 / tempo_bpm  # seconds per beat

        # Calculate total duration
        total_beats = float(sum(event.duration for event in rhythm))
        total_samples = int(total_beats * beat_duration * sr)

        audio = np.zeros(total_samples)

        sample_pos = 0
        for event in rhythm:
            duration_sec = float(event.duration) * beat_duration
            duration_samples = int(duration_sec * sr)

            if duration_samples <= 0:
                continue

            # Generate a short click at the start of each event
            click_samples = min(500, duration_samples // 10)
            if click_samples > 0:
                t = np.linspace(0, click_samples / sr, click_samples, endpoint=False)
                click = 0.8 * np.sin(2 * np.pi * click_freq * t) * np.exp(-t * sr * 5)

                end_click = min(sample_pos + click_samples, total_samples)
                actual_click_len = end_click - sample_pos
                if actual_click_len > 0:
                    audio[sample_pos:end_click] += click[:actual_click_len]

            sample_pos += duration_samples

        return audio

    def demonstrate_isomorphism(self) -> dict[str, Union[list, tuple]]:
        """Create a demo showing the 3:2 isomorphism in action.

        Returns a dict with melodic and rhythmic phrases.
        """
        # Create a simple melody with the perfect fifth (3:2)
        melody = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),      # Unison
            MelodicNote(ratio=Fraction(9, 8), duration=Fraction(1, 2)),      # Major second (faster)
            MelodicNote(ratio=Fraction(5, 4), duration=Fraction(1, 2)),      # Major third
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1)),      # PERFECT FIFTH! (longer)
            MelodicNote(ratio=Fraction(5, 3), duration=Fraction(1, 2)),      # Major sixth
            MelodicNote(ratio=Fraction(15, 8), duration=Fraction(1, 2)),     # Major seventh
            MelodicNote(ratio=Fraction(2, 1), duration=Fraction(1, 1)),      # Octave
        ]

        rhythm = self.melody_to_rhythm(melody)

        return {
            "melody": melody,
            "rhythm": rhythm,
            "isomorphism_note": "The perfect fifth (3:2) in pitch becomes hemiola (3 beats in 2) in rhythm!",
        }


# ─── Meantone Simulator: Quarter-Comma Meantone Tuning ───

class TemperamentType(Enum):
    """Types of historical temperaments."""
    QUARTER_COMMA_MEANTONE = "quarter_comma_meantone"
    PYTHAGOREAN = "pythagorean"
    EQUAL = "equal_temperament"


@dataclass
class MeantoneKey:
    """Information about a specific key in meantone temperament."""
    root_name: str
    root_ratio: Fraction
    major_third_quality: str  # "pure", "wide", "wolf"
    major_third_deviation_cents: float
    fifth_quality: str  # "narrow", "wide", "wolf"
    fifth_deviation_cents: float
    overall_color: str  # emotional description


class MeantoneSimulator:
    """Simulate quarter-comma meantone tuning and its key colors.

    Before equal temperament (~1750), each key had a distinct acoustic
    color because of temperament. Quarter-comma meantone narrowed each
    fifth by 1/4 of the syntonic comma (~5.4 cents), making major thirds
    PURE (5:4) in keys close to the home key.

    Remote keys had increasingly dissonant thirds — the "wolf intervals."
    This is why:
    - D major sounded triumphant (pure major third, bright)
    - C major sounded simple and pure (home key)
    - Remote keys sounded harsh or unusable
    """

    # Syntonic comma: 81:80, approximately 21.5 cents
    SYNTONIC_COMMA = Fraction(81, 80)
    SYNTONIC_COMMA_CENTS = ratio_to_cents(Fraction(81, 80))

    # Quarter-comma: 1/4 of syntonic comma per fifth
    QUARTER_COMMA_CENTS = SYNTONIC_COMMA_CENTS / 4

    def __init__(self, root: str = "C"):
        """Initialize the meantone simulator.

        Args:
            root: The root note (default: C for C-major meantone)
        """
        self.root = root.upper()
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.generate_tuning()

    def generate_tuning(self) -> None:
        """Generate the quarter-comma meantone tuning."""
        # Start from the root (C = 1:1)
        root_idx = self.note_names.index(self.root)

        # Generate by stacking fifths upward, each narrowed by 1/4 comma
        # In quarter-comma meantone, fifths are: 3:2 * (81/80)^(-1/4)
        # This makes the major third (four fifths up) pure 5:4

        self.ratios: dict[str, Fraction] = {}
        self.ratios[self.root] = Fraction(1, 1)

        # Stack fifths upward
        current_ratio = Fraction(1, 1)
        current_note = self.root

        for i in range(11):
            # Go up a fifth (7 semitones in 12-TET)
            note_idx = (self.note_names.index(current_note) + 7) % 12
            current_note = self.note_names[note_idx]

            # Apply quarter-comma narrowing
            # Pure fifth is 3:2, narrowed by 1/4 syntonic comma
            # Fraction ** Fraction returns float, so compute in float then convert back
            fifth_float = (3.0 / 2.0) / (81.0 / 80.0) ** 0.25
            current_ratio = Fraction(current_ratio) * Fraction.from_float(fifth_float).limit_denominator(10000)

            # Normalize to within one octave
            while current_ratio >= 2:
                current_ratio = current_ratio / 2
            while current_ratio < 1:
                current_ratio = current_ratio * 2

            self.ratios[current_note] = current_ratio

    def get_ratio(self, note: str) -> Fraction:
        """Get the ratio for a note name."""
        return self.ratios.get(note.upper(), Fraction(1, 1))

    def interval_ratio(self, note1: str, note2: str) -> Fraction:
        """Get the ratio between two notes."""
        r1 = self.get_ratio(note1)
        r2 = self.get_ratio(note2)
        ratio = r2 / r1

        # Normalize to [1, 2)
        while ratio >= 2:
            ratio = ratio / 2
        while ratio < 1:
            ratio = ratio * 2

        return ratio

    def analyze_key(self, key_name: str) -> MeantoneKey:
        """Analyze the quality of a specific key.

        Args:
            key_name: Name of the key (e.g., "D", "F#", "Bb")

        Returns:
            MeantoneKey with quality information
        """
        key = key_name.upper()

        # Get the major third (4 semitones up)
        root_idx = self.note_names.index(key)
        third_idx = (root_idx + 4) % 12
        third_name = self.note_names[third_idx]

        # Get the fifth (7 semitones up)
        fifth_idx = (root_idx + 7) % 12
        fifth_name = self.note_names[fifth_idx]

        # Calculate actual intervals
        major_third = self.interval_ratio(key, third_name)
        fifth = self.interval_ratio(key, fifth_name)

        # Compare to just intervals
        pure_major_third = Fraction(5, 4)
        pure_fifth = Fraction(3, 2)

        third_cents = ratio_to_cents(major_third)
        pure_third_cents = ratio_to_cents(pure_major_third)
        third_deviation = third_cents - pure_third_cents

        fifth_cents = ratio_to_cents(fifth)
        pure_fifth_cents = ratio_to_cents(pure_fifth)
        fifth_deviation = fifth_cents - pure_fifth_cents

        # Determine qualities
        if abs(third_deviation) < 2:
            third_quality = "pure"
        elif abs(third_deviation) < 15:
            third_quality = "wide" if third_deviation > 0 else "narrow"
        else:
            third_quality = "wolf"

        if abs(fifth_deviation) < 5:
            fifth_quality = "narrow"  # Meantone fifths are always slightly narrow
        elif abs(fifth_deviation) < 20:
            fifth_quality = "wide"
        else:
            fifth_quality = "wolf"

        # Determine overall color
        if third_quality == "pure":
            if key in ["D", "G", "A"]:
                color = "bright, triumphant, festive"
            elif key == self.root:
                color = "pure, simple, natural"
            else:
                color = "warm, resonant, sweet"
        elif third_quality == "wolf":
            color = "harsh, discordant, unusable"
        else:
            color = "slightly off, tense, distant"

        return MeantoneKey(
            root_name=key,
            root_ratio=self.get_ratio(key),
            major_third_quality=third_quality,
            major_third_deviation_cents=third_deviation,
            fifth_quality=fifth_quality,
            fifth_deviation_cents=fifth_deviation,
            overall_color=color,
        )

    def get_wolf_intervals(self) -> list[tuple[str, str, float]]:
        """Find all wolf intervals (highly dissonant) in this tuning.

        Returns:
            List of (note1, note2, deviation_cents) tuples
        """
        wolf_intervals = []

        for i, note1 in enumerate(self.note_names):
            for j, note2 in enumerate(self.note_names):
                if i >= j:
                    continue

                ratio = self.interval_ratio(note1, note2)
                pure_third = Fraction(5, 4)
                pure_fifth = Fraction(3, 2)

                deviation_third = abs(ratio_to_cents(ratio) - ratio_to_cents(pure_third))
                deviation_fifth = abs(ratio_to_cents(ratio) - ratio_to_cents(pure_fifth))

                # Wolf threshold: more than 20 cents deviation from pure intervals
                if deviation_third > 20 or deviation_fifth > 20:
                    wolf_intervals.append((note1, note2, max(deviation_third, deviation_fifth)))

        # Sort by deviation (most severe first)
        wolf_intervals.sort(key=lambda x: x[2], reverse=True)

        return wolf_intervals

    def render_chord(
        self,
        root: str,
        chord_type: str = "major",
        duration: float = 2.0,
        sr: int = 44100,
    ) -> np.ndarray:
        """Render a chord in meantone tuning.

        Args:
            root: Root note name
            chord_type: "major" or "minor"
            duration: Duration in seconds
            sr: Sample rate

        Returns:
            Audio array
        """
        root = root.upper()
        root_idx = self.note_names.index(root)

        # Determine chord notes
        if chord_type == "major":
            # Root, major third, perfect fifth
            third_idx = (root_idx + 4) % 12
            fifth_idx = (root_idx + 7) % 12
        else:  # minor
            # Root, minor third, perfect fifth
            third_idx = (root_idx + 3) % 12
            fifth_idx = (root_idx + 7) % 12

        notes = [root, self.note_names[third_idx], self.note_names[fifth_idx]]

        # Generate audio
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)

        audio = np.zeros_like(t)
        fundamental = 261.63  # C4 (if root is C)

        for i, note in enumerate(notes):
            ratio = self.get_ratio(note)
            freq = fundamental * float(ratio) * (2 ** ((self.note_names.index(note) - self.note_names.index(self.root)) // 12))

            # Adjust octave
            octave_diff = (self.note_names.index(note) - self.note_names.index(self.root)) // 12
            freq = freq * (2 ** octave_diff)

            wave = 0.3 * np.sin(2 * np.pi * freq * t)

            # Apply envelope
            envelope = np.ones_like(t)
            fade_len = int(0.05 * sr)
            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)
            wave = wave * envelope

            audio += wave

        return audio

    def compare_keys(
        self,
        key1: str,
        key2: str,
        sr: int = 44100,
        chord_duration: float = 2.0,
    ) -> tuple[np.ndarray, str]:
        """Render the same chord progression in two different keys.

        This demonstrates the color difference between keys.

        Returns:
            (audio, description) tuple
        """
        # Play I-IV-V-I progression in each key
        key1_info = self.analyze_key(key1)
        key2_info = self.analyze_key(key2)

        # Render key 1
        audio1 = self.render_chord(key1, "major", chord_duration, sr)
        audio1 += self.render_chord(self._note_up(key1, 5), "major", chord_duration, sr)  # IV
        audio1 += self.render_chord(self._note_up(key1, 7), "major", chord_duration, sr)  # V
        audio1 += self.render_chord(key1, "major", chord_duration, sr)  # I

        # Add silence between keys
        silence = np.zeros(int(0.5 * sr))
        audio1 = np.concatenate([audio1, silence])

        # Render key 2
        audio2 = self.render_chord(key2, "major", chord_duration, sr)
        audio2 += self.render_chord(self._note_up(key2, 5), "major", chord_duration, sr)  # IV
        audio2 += self.render_chord(self._note_up(key2, 7), "major", chord_duration, sr)  # V
        audio2 += self.render_chord(key2, "major", chord_duration, sr)  # I

        audio = np.concatenate([audio1, audio2])

        description = (
            f"{key1} major: {key1_info.overall_color} "
            f"(third: {key1_info.major_third_quality}, "
            f"{key1_info.major_third_deviation_cents:+.1f}¢)\n"
            f"{key2} major: {key2_info.overall_color} "
            f"(third: {key2_info.major_third_quality}, "
            f"{key2_info.major_third_deviation_cents:+.1f}¢)"
        )

        return audio, description

    def _note_up(self, note: str, semitones: int) -> str:
        """Get the note name `semitones` above `note`."""
        note_upper = note.upper()
        # Handle flat notation
        if note_upper.endswith('B'):
            note_upper = note_upper.replace('B', '#')
        idx = (self.note_names.index(note_upper) + semitones) % 12
        return self.note_names[idx]


# ─── Nancarrow Study 37: 12-Voice Canon with Just-Intonation Tempos ───

@dataclass
class NancarrowVoice:
    """A single voice in Nancarrow's Study 37."""
    voice_number: int
    tempo_ratio: Fraction      # Ratio relative to fundamental tempo
    base_midi_pitch: int = 60  # Base pitch
    melody: list[int] = field(default_factory=list)  # MIDI notes

    def tempo_bpm(self, fundamental_bpm: float = 120.0) -> float:
        """Calculate this voice's tempo in BPM."""
        return fundamental_bpm * float(self.tempo_ratio)

    def render(
        self,
        fundamental_bpm: float = 120.0,
        duration_beats: float = 16.0,
        sr: int = 44100,
    ) -> np.ndarray:
        """Render this voice as audio.

        Args:
            fundamental_bpm: Base tempo for ratio 1/1
            duration_beats: Duration in fundamental beats
            sr: Sample rate

        Returns:
            Audio array
        """
        voice_bpm = self.tempo_bpm(fundamental_bpm)
        beat_duration = 60.0 / voice_bpm
        total_duration = duration_beats * beat_duration

        total_samples = int(total_duration * sr)
        audio = np.zeros(total_samples)

        # Simple melody: arpeggiate through scale degrees
        if not self.melody:
            # Default: simple ascending pattern
            self.melody = [0, 2, 4, 5, 7, 9, 11, 12]  # Major scale

        time = 0.0
        note_idx = 0

        while time < total_duration:
            # Get current note
            degree = self.melody[note_idx % len(self.melody)]
            midi_pitch = self.base_midi_pitch + degree

            # Frequency
            freq = 440.0 * (2 ** ((midi_pitch - 69) / 12))

            # Note duration (1 beat)
            note_duration = beat_duration
            note_samples = int(note_duration * sr)

            if time + note_samples > total_samples:
                note_samples = int(total_samples - time)
                if note_samples <= 0:
                    break

            # Generate wave
            t = np.linspace(0, note_samples / sr, note_samples, endpoint=False)
            wave = 0.15 * np.sin(2 * np.pi * freq * t)

            # Envelope
            envelope = np.ones_like(wave)
            fade_len = min(100, note_samples // 10)
            if fade_len > 0:
                envelope[:fade_len] = np.linspace(0, 1, fade_len)
                envelope[-fade_len:] = np.linspace(1, 0, fade_len)
            wave = wave * envelope

            # Mix into audio
            start_sample = int(time * sr)
            end_sample = start_sample + note_samples
            if end_sample > total_samples:
                end_sample = total_samples

            actual_len = end_sample - start_sample
            if actual_len > 0:
                audio[start_sample:end_sample] += wave[:actual_len]

            time += note_duration
            note_idx += 1

        return audio


class NancarrowStudy37:
    """Renderer for Nancarrow's Study 37 - 12-voice polytemporal canon.

    Study 37 is Nancarrow's masterpiece of the pitch-rhythm analogy.
    Each voice moves at a tempo corresponding to a just-intonation interval:

    Voice  1: MM 150      = 1/1   (unison)
    Voice  2: MM 160 5/7  = 15/14 (semitone)
    Voice  3: MM 168 3/4  = 9/8   (major second)
    Voice  4: MM 180      = 6/5   (minor third)
    Voice  5: MM 187 1/2  = 5/4   (major third)
    Voice  6: MM 200      = 4/3   (perfect fourth)
    Voice  7: MM 210      = 7/5   (tritone)
    Voice  8: MM 225      = 3/2   (PERFECT FIFTH)
    Voice  9: MM 240      = 8/5   (minor sixth)
    Voice 10: MM 250      = 5/3   (major sixth)
    Voice 11: MM 262 1/2  = 7/4   (minor seventh)
    Voice 12: MM 281 1/4  = 15/8  (major seventh)

    Voice 8 moves at tempo ratio 3/2 — the perfect fifth!
    This is literally the perfect fifth existing simultaneously as
    pitch relationship AND tempo relationship.

    When voices align in time, they create "temporal consonance" —
    moments where the rhythm resolves, analogous to harmonic resolution.
    """

    # Nancarrow's actual tempo ratios from Study 37
    NANCARROW_RATIOS = [
        Fraction(1, 1),       # Unison
        Fraction(15, 14),     # Semitone
        Fraction(9, 8),       # Major second
        Fraction(6, 5),       # Minor third
        Fraction(5, 4),       # Major third
        Fraction(4, 3),       # Perfect fourth
        Fraction(7, 5),       # Tritone
        Fraction(3, 2),       # PERFECT FIFTH!
        Fraction(8, 5),       # Minor sixth
        Fraction(5, 3),       # Major sixth
        Fraction(7, 4),       # Minor seventh
        Fraction(15, 8),      # Major seventh
    ]

    def __init__(self, fundamental_bpm: float = 150.0):
        """Initialize the Study 37 renderer.

        Args:
            fundamental_bpm: Base tempo (Nancarrow used MM 150 for voice 1)
        """
        self.fundamental_bpm = fundamental_bpm
        self.voices = self._create_voices()

    def _create_voices(self) -> list[NancarrowVoice]:
        """Create all 12 voices with their tempo ratios."""
        voices = []
        for i, ratio in enumerate(self.NANCARROW_RATIOS):
            # Each voice has a different base pitch for variety
            base_pitch = 60 + (i % 12)  # Spread across an octave
            voices.append(NancarrowVoice(
                voice_number=i + 1,
                tempo_ratio=ratio,
                base_midi_pitch=base_pitch,
            ))
        return voices

    def find_alignment_points(
        self,
        duration_beats: float = 16.0,
    ) -> list[tuple[float, list[int]]]:
        """Find when voices align (create temporal consonance).

        Alignment occurs when multiple voices are at the same beat position.
        These are moments of "temporal consonance" — the rhythm resolves.

        Args:
            duration_beats: Duration to analyze (in fundamental beats)

        Returns:
            List of (time, voice_indices) tuples
        """
        alignments = []

        # Check each fundamental beat
        for t in np.arange(0, duration_beats, 0.1):
            aligned_voices = []
            for i, voice in enumerate(self.voices):
                # Calculate this voice's beat position
                voice_beat = t * float(voice.tempo_ratio)
                # Check if it's close to a beat boundary
                if abs(voice_beat - round(voice_beat)) < 0.05:
                    aligned_voices.append(i)

            # If 3 or more voices align, record it
            if len(aligned_voices) >= 3:
                alignments.append((t, aligned_voices))

        return alignments

    def render(
        self,
        duration_beats: float = 16.0,
        sr: int = 44100,
        voice_subset: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Render the full 12-voice canon.

        Args:
            duration_beats: Duration in fundamental beats
            sr: Sample rate
            voice_subset: If provided, only render these voice indices (0-11)

        Returns:
            Audio array with all voices mixed
        """
        if voice_subset is None:
            voices_to_render = self.voices
        else:
            voices_to_render = [self.voices[i] for i in voice_subset]

        # Find the maximum duration (fastest voice)
        max_duration = 0.0
        for voice in voices_to_render:
            voice_duration = duration_beats / float(voice.tempo_ratio)
            max_duration = max(max_duration, voice_duration)

        total_samples = int(max_duration * sr)
        audio = np.zeros(total_samples)

        # Render each voice
        for voice in voices_to_render:
            voice_audio = voice.render(self.fundamental_bpm, duration_beats, sr)

            # Mix into master (normalize to prevent clipping)
            voice_samples = min(len(voice_audio), total_samples)
            if voice_samples > 0:
                audio[:voice_samples] += voice_audio[:voice_samples]

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8

        return audio

    def analyze_temporal_consonance(
        self,
        duration_beats: float = 16.0,
    ) -> dict[str, Union[list, str]]:
        """Analyze the temporal consonance patterns in the canon.

        Returns:
            Dict with alignment points and analysis
        """
        alignments = self.find_alignment_points(duration_beats)

        # Count how often each voice participates in alignments
        voice_alignment_count = [0] * len(self.voices)
        for _, voices in alignments:
            for v in voices:
                voice_alignment_count[v] += 1

        # Find the most consonant voice (participates in most alignments)
        most_consonant = voice_alignment_count.index(max(voice_alignment_count))

        analysis = {
            "alignments": alignments,
            "voice_alignment_count": voice_alignment_count,
            "most_consonant_voice": most_consonant,
            "most_consonant_ratio": self.voices[most_consonant].tempo_ratio,
            "insight": (
                f"Voice {most_consonant + 1} (ratio {self.voices[most_consonant].tempo_ratio}) "
                f"participates in {voice_alignment_count[most_consonant]} alignments. "
                f"This is the 'temporal tonic' — the voice that most frequently creates resolution."
            ),
        }

        return analysis


# ─── Utility Functions ───

def save_audio(audio: np.ndarray, filename: str, sr: int = 44100) -> None:
    """Save audio to a WAV file.

    Requires scipy or soundfile. If not available, prints a warning.
    """
    try:
        import soundfile as sf
        sf.write(filename, audio, sr)
        print(f"Saved audio to {filename}")
    except ImportError:
        try:
            from scipy.io import wavfile
            wavfile.write(filename, sr, audio)
            print(f"Saved audio to {filename}")
        except ImportError:
            print("Warning: Neither soundfile nor scipy available. Cannot save audio.")


def plot_isomorphism(melody: list[MelodicNote], rhythm: list[RhythmicEvent]) -> None:
    """Create a visualization of the pitch-rhythm isomorphism.

    Requires matplotlib. If not available, prints a warning.
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot melody (pitch domain)
        times_pitch = np.cumsum([0] + [float(n.duration) for n in melody[:-1]])
        pitches = [float(n.ratio) for n in melody]
        ax1.step(times_pitch, pitches, where='post', linewidth=2)
        ax1.set_ylabel('Pitch Ratio')
        ax1.set_title('Melodic Phrase (Pitch Domain)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.8, 2.2)

        # Add annotations
        for i, (t, p) in enumerate(zip(times_pitch, pitches)):
            ax1.annotate(f'{melody[i].ratio}', (t, p), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

        # Plot rhythm (time domain)
        times_rhythm = np.cumsum([0] + [float(e.duration) for e in rhythm[:-1]])
        durations = [float(e.duration) for e in rhythm]
        ax2.bar(range(len(durations)), durations, alpha=0.7)
        ax2.set_ylabel('Duration (beats)')
        ax2.set_xlabel('Event Number')
        ax2.set_title('Rhythmic Phrase (Time Domain)')
        ax2.set_xticks(range(len(durations)))
        ax2.set_xticklabels([f'{rhythm[i].duration}' for i in range(len(durations))])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/tmp/publish/constraint-synth/isomorphism_plot.png', dpi=150)
        print("Saved isomorphism plot to isomorphism_plot.png")

    except ImportError:
        print("Warning: matplotlib not available. Cannot plot.")


# ─── Demo Function ───

def demo_three_halves() -> None:
    """Run a comprehensive demo of all three components."""
    print("=" * 70)
    print("THREE HALVES: The 3:2 Isomorphism Between Pitch and Rhythm")
    print("=" * 70)

    # 1. ThreeHalves Demo
    print("\n1. THREE HALVES ISOMORPHISM")
    print("-" * 70)
    th = ThreeHalves()
    demo = th.demonstrate_isomorphism()

    print(f"Melody ({len(demo['melody'])} notes):")
    for i, note in enumerate(demo['melody']):
        ratio_str = str(note.ratio).rjust(6)
        duration_str = str(note.duration).rjust(6)
        print(f"  {i+1}. Ratio {ratio_str} = {ratio_to_cents(note.ratio):6.1f}¢ "
              f"(MIDI {note.midi_pitch}), duration {duration_str}")

    print(f"\nRhythm ({len(demo['rhythm'])} events):")
    for i, event in enumerate(demo['rhythm']):
        duration_str = str(event.duration).rjust(6)
        print(f"  {i+1}. Duration {duration_str} beats "
              f"= {float(event.duration):.2f} beats")

    print(f"\n{demo['isomorphism_note']}")

    # 2. MeantoneSimulator Demo
    print("\n2. MEANTONE TEMPERAMENT SIMULATOR")
    print("-" * 70)
    meantone = MeantoneSimulator(root="C")

    # Analyze a few keys
    keys_to_analyze = ["C", "D", "G", "F#", "A#"]
    for key in keys_to_analyze:
        info = meantone.analyze_key(key)
        print(f"{key} major:")
        print(f"  Color: {info.overall_color}")
        print(f"  Major third: {info.major_third_quality} "
              f"({info.major_third_deviation_cents:+.1f}¢ from pure)")
        print(f"  Fifth: {info.fifth_quality} "
              f"({info.fifth_deviation_cents:+.1f}¢ from pure)")

    # Find wolf intervals
    wolves = meantone.get_wolf_intervals()
    if wolves:
        print(f"\nWolf intervals (top 5):")
        for note1, note2, deviation in wolves[:5]:
            print(f"  {note1}-{note2}: {deviation:.1f}¢ deviation")

    # 3. NancarrowStudy37 Demo
    print("\n3. NANCARROW STUDY 37: 12-Voice Canon")
    print("-" * 70)
    study37 = NancarrowStudy37(fundamental_bpm=150.0)

    print("Voice tempo ratios:")
    for voice in study37.voices[:8]:  # Show first 8
        ratio_str = str(voice.tempo_ratio).rjust(7)
        print(f"  Voice {voice.voice_number:2d}: {ratio_str} "
              f"= {float(voice.tempo_ratio):.4f} = {voice.tempo_bpm(study37.fundamental_bpm):.1f} BPM")

    # Analyze temporal consonance
    analysis = study37.analyze_temporal_consonance(duration_beats=16.0)
    print(f"\nTemporal consonance analysis:")
    print(f"  Found {len(analysis['alignments'])} alignment points")
    print(f"  Most consonant voice: {analysis['most_consonant_voice'] + 1} "
          f"(ratio {analysis['most_consonant_ratio']})")
    print(f"  {analysis['insight']}")

    print("\n" + "=" * 70)
    print("This is art disguised as engineering.")
    print("=" * 70)


if __name__ == "__main__":
    demo_three_halves()