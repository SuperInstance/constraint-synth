"""HarmonicSynthesizer — generating constraint-satisfying musical sequences.

Produces melodies, chord progressions, and harmonies where every note
respects a given scale/constraint set. Uses the lattice and scale
system to ensure all generated music is theoretically sound.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

from .scales import (
    SCALES,
    TraditionScale,
    ratio_to_cents,
    ratio_to_semitones,
    consonance_score,
    tenney_height,
    UNISON,
    OCTAVE,
    PERFECT_FIFTH,
    PERFECT_FOURTH,
    MAJOR_THIRD,
    MINOR_THIRD,
)


def _midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _ratio_to_midi_offset(ratio: Fraction) -> float:
    """Convert a just-intonation ratio to semitone offset from root."""
    return ratio_to_semitones(ratio)


@dataclass
class HarmonicVoice:
    """A single voice in a harmonic texture.

    Attributes:
        midi_root: The MIDI note number of the voice's root.
        velocity: Loudness (0-127).
        channel: Logical MIDI channel (for polyphonic tracking).
    """
    midi_root: int
    velocity: int = 100
    channel: int = 0

    def frequency(self) -> float:
        return _midi_to_freq(self.midi_root)


@dataclass
class Note:
    """A single note event in a sequence.

    Attributes:
        midi_pitch: MIDI note number (0-127).
        velocity: Loudness (0-127).
        duration_beats: Length in beats (1.0 = quarter note).
        offset_beats: Start time in beats from sequence beginning.
    """
    midi_pitch: int
    velocity: int = 100
    duration_beats: float = 1.0
    offset_beats: float = 0.0


@dataclass
class HarmonicSequence:
    """A sequence of constraint-satisfying notes.

    Attributes:
        notes: Ordered list of Note events.
        scale_name: Name of the scale used to generate this sequence.
        tempo_bpm: Tempo in beats per minute.
    """
    notes: list[Note] = field(default_factory=list)
    scale_name: str = "major"
    tempo_bpm: float = 120.0

    def total_duration_beats(self) -> float:
        """Total length of the sequence in beats."""
        if not self.notes:
            return 0.0
        return max(n.offset_beats + n.duration_beats for n in self.notes)

    def total_duration_seconds(self) -> float:
        """Total length in seconds."""
        return self.total_duration_beats() * (60.0 / self.tempo_bpm)

    def pitch_classes(self) -> list[int]:
        """Return all pitch classes (0-11) used in the sequence."""
        return sorted(set(n.midi_pitch % 12 for n in self.notes))

    def transpose(self, semitones: int) -> HarmonicSequence:
        """Return a new sequence transposed by the given number of semitones."""
        return HarmonicSequence(
            notes=[
                Note(
                    midi_pitch=n.midi_pitch + semitones,
                    velocity=n.velocity,
                    duration_beats=n.duration_beats,
                    offset_beats=n.offset_beats,
                )
                for n in self.notes
            ],
            scale_name=self.scale_name,
            tempo_bpm=self.tempo_bpm,
        )

    def invert_around(self, pivot: int) -> HarmonicSequence:
        """Invert all intervals around a pivot MIDI note."""
        return HarmonicSequence(
            notes=[
                Note(
                    midi_pitch=2 * pivot - n.midi_pitch,
                    velocity=n.velocity,
                    duration_beats=n.duration_beats,
                    offset_beats=n.offset_beats,
                )
                for n in self.notes
            ],
            scale_name=self.scale_name,
            tempo_bpm=self.tempo_bpm,
        )


@dataclass
class HarmonicSynthesizer:
    """Generates constraint-satisfying musical sequences.

    The synthesizer produces melodies, chord progressions, and harmonic
    textures where every pitch belongs to the chosen scale. It uses
    just-intonation ratios internally for accurate interval quality.

    Attributes:
        scale_name: Name of the scale from the SCALES registry.
        root_midi: MIDI note number of the tonic / root.
        tempo_bpm: Tempo in beats per minute.
        default_velocity: Default note velocity.
        rng_seed: Random seed for reproducibility (None = non-deterministic).
    """
    scale_name: str = "major"
    root_midi: int = 60  # Middle C
    tempo_bpm: float = 120.0
    default_velocity: int = 100
    rng_seed: Optional[int] = None

    def __post_init__(self):
        if self.scale_name not in SCALES:
            raise ValueError(
                f"Unknown scale '{self.scale_name}'. "
                f"Available: {list(SCALES.keys())}"
            )
        self._rng = random.Random(self.rng_seed)
        self._scale: TraditionScale = SCALES[self.scale_name]

    # ── Scale degree → MIDI pitch ──────────────────────────────────

    def _scale_degrees_to_midi(self) -> list[int]:
        """Map each scale degree (including root and octave) to MIDI pitches.

        Returns a list starting with the root, then each scale degree,
        then the octave, all as MIDI note numbers.
        """
        root = self.root_midi
        pitches = [root]
        for ratio in self._scale.intervals:
            semitones = round(ratio_to_semitones(ratio))
            pitches.append(root + semitones)
        pitches.append(root + 12)  # octave
        return pitches

    def scale_pitches(self, octave_range: int = 1) -> list[int]:
        """Get all valid MIDI pitches across the given octave range.

        Args:
            octave_range: Number of octaves above (and below) root to include.

        Returns:
            Sorted list of MIDI pitches that belong to this scale.
        """
        base = self._scale_degrees_to_midi()
        pitches = []
        for octave_shift in range(-octave_range, octave_range + 1):
            for p in base:
                candidate = p + 12 * octave_shift
                if 0 <= candidate <= 127:
                    pitches.append(candidate)
        return sorted(set(pitches))

    # ── Melody Generation ──────────────────────────────────────────

    def generate_melody(
        self,
        num_notes: int = 8,
        max_interval: int = 5,
        duration_options: Optional[list[float]] = None,
    ) -> HarmonicSequence:
        """Generate a constraint-satisfying melody.

        Each note belongs to the chosen scale and no interval between
        consecutive notes exceeds `max_interval` semitones.

        Args:
            num_notes: Number of notes to generate.
            max_interval: Maximum semitone jump between consecutive notes.
            duration_options: Allowed beat durations (default: [0.5, 1.0, 2.0]).

        Returns:
            A HarmonicSequence containing the generated melody.
        """
        if duration_options is None:
            duration_options = [0.5, 1.0, 2.0]

        valid_pitches = self.scale_pitches(octave_range=1)
        notes: list[Note] = []
        current_pitch = self.root_midi
        offset = 0.0

        for i in range(num_notes):
            # Filter pitches within max_interval of current
            candidates = [
                p for p in valid_pitches
                if abs(p - current_pitch) <= max_interval
            ]
            if not candidates:
                candidates = [current_pitch]

            # Weight toward stepwise motion (smaller intervals preferred)
            weights = [1.0 / (1 + abs(p - current_pitch)) for p in candidates]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            chosen = self._rng.choices(candidates, weights=weights, k=1)[0]
            dur = self._rng.choice(duration_options)

            notes.append(Note(
                midi_pitch=chosen,
                velocity=self.default_velocity,
                duration_beats=dur,
                offset_beats=offset,
            ))

            current_pitch = chosen
            offset += dur

        return HarmonicSequence(
            notes=notes,
            scale_name=self.scale_name,
            tempo_bpm=self.tempo_bpm,
        )

    # ── Chord Generation ───────────────────────────────────────────

    def generate_chord(
        self,
        degree: int = 1,
        voicing: str = "triad",
        inversion: int = 0,
    ) -> list[Note]:
        """Generate a chord on the given scale degree.

        Args:
            degree: Scale degree (1 = tonic).
            voicing: "triad" (3 notes) or "seventh" (4 notes).
            inversion: 0 = root position, 1 = first inversion, etc.

        Returns:
            List of Note objects forming the chord (all at offset 0).
        """
        pitches_in_octave = self._scale_degrees_to_midi()
        # Remove octave duplicate for degree indexing
        scale_pitches = pitches_in_octave[:-1]

        # Build chord tones by stacking thirds
        root_idx = (degree - 1) % len(scale_pitches)
        chord_indices = [root_idx]
        if voicing in ("triad", "seventh"):
            chord_indices.append((root_idx + 2) % len(scale_pitches))
        if voicing == "seventh":
            chord_indices.append((root_idx + 4) % len(scale_pitches))
        chord_indices.append((root_idx + 2) % len(scale_pitches))
        # For triad: root, 3rd, 5th
        # For seventh: root, 3rd, 5th, 7th
        if voicing == "triad":
            chord_degrees = [0, 2, 4]
        else:
            chord_degrees = [0, 2, 4, 6]

        chord_pitches = []
        for d in chord_degrees:
            idx = (root_idx + d) % len(scale_pitches)
            octave_shift = (root_idx + d) // len(scale_pitches)
            pitch = scale_pitches[idx] + 12 * octave_shift
            chord_pitches.append(pitch)

        # Apply inversion: raise the lowest `inversion` notes by an octave
        for inv in range(min(inversion, len(chord_pitches))):
            chord_pitches[inv] += 12

        # Sort for readability
        chord_pitches.sort()

        return [
            Note(midi_pitch=p, velocity=self.default_velocity,
                 duration_beats=1.0, offset_beats=0.0)
            for p in chord_pitches
        ]

    def generate_chord_progression(
        self,
        degrees: Optional[list[int]] = None,
        beats_per_chord: float = 4.0,
        voicing: str = "triad",
    ) -> HarmonicSequence:
        """Generate a chord progression from scale degrees.

        Args:
            degrees: Scale degrees for each chord (default: [1, 4, 5, 1]).
            beats_per_chord: Duration of each chord in beats.
            voicing: "triad" or "seventh".

        Returns:
            HarmonicSequence with all chord notes.
        """
        if degrees is None:
            degrees = [1, 4, 5, 1]

        all_notes: list[Note] = []
        offset = 0.0

        for deg in degrees:
            chord = self.generate_chord(degree=deg, voicing=voicing)
            for note in chord:
                all_notes.append(Note(
                    midi_pitch=note.midi_pitch,
                    velocity=note.velocity,
                    duration_beats=beats_per_chord,
                    offset_beats=offset,
                ))
            offset += beats_per_chord

        return HarmonicSequence(
            notes=all_notes,
            scale_name=self.scale_name,
            tempo_bpm=self.tempo_bpm,
        )

    # ── Counterpoint / Voice Leading ───────────────────────────────

    def generate_two_voice(
        self,
        num_bars: int = 4,
        beats_per_bar: float = 4.0,
    ) -> HarmonicSequence:
        """Generate a simple two-voice contrapuntal texture.

        The upper voice moves stepwise; the lower voice provides
        harmonic support with consonant intervals (3rds, 5ths, 6ths).

        Args:
            num_bars: Number of bars to generate.
            beats_per_bar: Beats per bar.

        Returns:
            HarmonicSequence with two interleaved voices.
        """
        valid_pitches = self.scale_pitches(octave_range=2)
        upper_root = self.root_midi + 12  # one octave up for cantus firmus
        lower_root = self.root_midi

        notes: list[Note] = []
        upper_pitch = upper_root
        lower_pitch = lower_root
        offset = 0.0
        total_beats = num_bars * beats_per_bar
        beat = 0.0

        while beat < total_beats:
            # Upper voice: stepwise motion
            step_candidates = [
                p for p in valid_pitches
                if abs(p - upper_pitch) <= 2 and p != upper_pitch
            ]
            if step_candidates:
                upper_pitch = self._rng.choice(step_candidates)

            # Lower voice: consonant with upper (3rd, 5th, 6th, or octave below)
            consonant_targets = [
                upper_pitch - 3,   # minor 3rd below
                upper_pitch - 4,   # major 3rd below
                upper_pitch - 5,   # perfect 4th below
                upper_pitch - 7,   # perfect 5th below
                upper_pitch - 9,   # major 6th below
                upper_pitch - 12,  # octave below
            ]
            # Pick a consonant target that's in the scale, or keep current
            for target in self._rng.sample(consonant_targets, len(consonant_targets)):
                if target in valid_pitches:
                    lower_pitch = target
                    break

            dur = 1.0  # whole-beat notes
            notes.append(Note(
                midi_pitch=upper_pitch,
                velocity=self.default_velocity,
                duration_beats=dur,
                offset_beats=offset,
            ))
            notes.append(Note(
                midi_pitch=lower_pitch,
                velocity=int(self.default_velocity * 0.8),
                duration_beats=dur,
                offset_beats=offset,
            ))

            offset += dur
            beat += dur

        return HarmonicSequence(
            notes=notes,
            scale_name=self.scale_name,
            tempo_bpm=self.tempo_bpm,
        )

    # ── Sequence Manipulation ───────────────────────────────────────

    def rhythmic_variation(
        self,
        sequence: HarmonicSequence,
    ) -> HarmonicSequence:
        """Create a rhythmic variation of an existing sequence.

        Keeps the same pitches but varies durations and adds rests.
        """
        durations = [0.25, 0.5, 0.5, 1.0, 1.0, 2.0]
        offset = 0.0
        new_notes: list[Note] = []

        for note in sequence.notes:
            new_dur = self._rng.choice(durations)
            # Occasionally insert a rest (skip the note)
            if self._rng.random() < 0.15:
                offset += new_dur
                continue
            new_notes.append(Note(
                midi_pitch=note.midi_pitch,
                velocity=note.velocity,
                duration_beats=new_dur,
                offset_beats=offset,
            ))
            offset += new_dur

        return HarmonicSequence(
            notes=new_notes,
            scale_name=sequence.scale_name,
            tempo_bpm=sequence.tempo_bpm,
        )

    def retrograde(self, sequence: HarmonicSequence) -> HarmonicSequence:
        """Reverse the order of notes in a sequence (retrograde).

        Preserves relative timing by mirroring offsets.
        """
        if not sequence.notes:
            return HarmonicSequence(
                scale_name=sequence.scale_name,
                tempo_bpm=sequence.tempo_bpm,
            )
        total = sequence.total_duration_beats()
        return HarmonicSequence(
            notes=[
                Note(
                    midi_pitch=n.midi_pitch,
                    velocity=n.velocity,
                    duration_beats=n.duration_beats,
                    offset_beats=total - n.offset_beats - n.duration_beats,
                )
                for n in reversed(sequence.notes)
            ],
            scale_name=sequence.scale_name,
            tempo_bpm=sequence.tempo_bpm,
        )
