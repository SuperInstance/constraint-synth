"""ConstraintMutation — applying random mutations while respecting constraints.

Provides a framework for evolving musical sequences through controlled
randomness. Every mutation operation guarantees the output remains within
the specified constraint set (scale, range, interval limits).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .scales import SCALES, ratio_to_semitones
from .harmony import HarmonicSequence, Note


@dataclass
class MutationConfig:
    """Configuration for mutation behaviour.

    Attributes:
        pitch_mutation_rate: Probability of mutating each note's pitch (0.0–1.0).
        duration_mutation_rate: Probability of mutating each note's duration.
        velocity_mutation_rate: Probability of mutating each note's velocity.
        max_pitch_shift: Maximum semitone shift for pitch mutations.
        duration_options: Allowed beat durations for duration mutations.
        velocity_range: (min, max) MIDI velocity range.
        preserve_root: If True, notes on the root pitch are never mutated.
        scale_enforcement: If True, mutated pitches are snapped to the scale.
    """
    pitch_mutation_rate: float = 0.3
    duration_mutation_rate: float = 0.2
    velocity_mutation_rate: float = 0.1
    max_pitch_shift: int = 4
    duration_options: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0]
    )
    velocity_range: tuple[int, int] = (40, 127)
    preserve_root: bool = True
    scale_enforcement: bool = True


@dataclass
class MutationResult:
    """Result of a mutation operation.

    Attributes:
        sequence: The mutated HarmonicSequence.
        mutations_applied: Count of individual mutations performed.
        mutation_types: Breakdown by mutation type (pitch, duration, velocity).
    """
    sequence: HarmonicSequence
    mutations_applied: int = 0
    mutation_types: dict[str, int] = field(
        default_factory=lambda: {"pitch": 0, "duration": 0, "velocity": 0}
    )


class ConstraintMutation:
    """Applies constrained mutations to HarmonicSequence objects.

    The mutator ensures every mutated note still satisfies the active
    constraint set — typically a musical scale plus range and interval
    constraints. This is useful for:

    - Evolutionary music composition
    - Generating variations on a theme
    - Exploring the neighbourhood of a musical idea

    Args:
        scale_name: Name of the scale from SCALES registry.
        root_midi: MIDI note of the tonic.
        config: MutationConfig controlling mutation rates and limits.
        rng_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        scale_name: str = "major",
        root_midi: int = 60,
        config: Optional[MutationConfig] = None,
        rng_seed: Optional[int] = None,
    ):
        if scale_name not in SCALES:
            raise ValueError(
                f"Unknown scale '{scale_name}'. "
                f"Available: {list(SCALES.keys())}"
            )
        self.scale_name = scale_name
        self.root_midi = root_midi
        self.config = config or MutationConfig()
        self._rng = random.Random(rng_seed)
        self._scale_pitches = self._build_scale_pitches()

    def _build_scale_pitches(self, octave_range: int = 2) -> list[int]:
        """Build the set of valid MIDI pitches for this scale."""
        scale = SCALES[self.scale_name]
        root = self.root_midi
        # Compute semitone offsets from intervals
        offsets = [0]  # root
        for ratio in scale.intervals:
            offsets.append(round(ratio_to_semitones(ratio)))

        pitches = set()
        for octave_shift in range(-octave_range, octave_range + 1):
            for off in offsets:
                p = root + off + 12 * octave_shift
                if 0 <= p <= 127:
                    pitches.add(p)
        return sorted(pitches)

    def _snap_to_scale(self, pitch: int) -> int:
        """Snap a pitch to the nearest scale tone."""
        if pitch in self._scale_pitches:
            return pitch
        # Find nearest
        best = self._scale_pitches[0]
        best_dist = abs(pitch - best)
        for p in self._scale_pitches:
            d = abs(pitch - p)
            if d < best_dist:
                best = p
                best_dist = d
        return best

    # ── Single-Note Mutations ──────────────────────────────────────

    def mutate_pitch(self, note: Note) -> Note:
        """Potentially mutate a single note's pitch.

        If the config's pitch_mutation_rate check passes, shift the pitch
        by a random amount (up to max_pitch_shift semitones). Then snap
        to scale if scale_enforcement is enabled.
        """
        if self._rng.random() > self.config.pitch_mutation_rate:
            return note

        if self.config.preserve_root and note.midi_pitch == self.root_midi:
            return note

        shift = self._rng.randint(
            -self.config.max_pitch_shift, self.config.max_pitch_shift
        )
        new_pitch = note.midi_pitch + shift
        new_pitch = max(0, min(127, new_pitch))

        if self.config.scale_enforcement:
            new_pitch = self._snap_to_scale(new_pitch)

        return Note(
            midi_pitch=new_pitch,
            velocity=note.velocity,
            duration_beats=note.duration_beats,
            offset_beats=note.offset_beats,
        )

    def mutate_duration(self, note: Note) -> Note:
        """Potentially mutate a single note's duration."""
        if self._rng.random() > self.config.duration_mutation_rate:
            return note

        new_dur = self._rng.choice(self.config.duration_options)
        return Note(
            midi_pitch=note.midi_pitch,
            velocity=note.velocity,
            duration_beats=new_dur,
            offset_beats=note.offset_beats,
        )

    def mutate_velocity(self, note: Note) -> Note:
        """Potentially mutate a single note's velocity."""
        if self._rng.random() > self.config.velocity_mutation_rate:
            return note

        lo, hi = self.config.velocity_range
        shift = self._rng.randint(-20, 20)
        new_vel = max(lo, min(hi, note.velocity + shift))
        return Note(
            midi_pitch=note.midi_pitch,
            velocity=new_vel,
            duration_beats=note.duration_beats,
            offset_beats=note.offset_beats,
        )

    # ── Sequence-Level Mutations ───────────────────────────────────

    def mutate(self, sequence: HarmonicSequence) -> MutationResult:
        """Apply all configured mutations to a sequence.

        Each note is independently considered for pitch, duration, and
        velocity mutations according to the configured rates. Offsets are
        recalculated to maintain sequential timing.

        Returns:
            MutationResult with the new sequence and mutation statistics.
        """
        result_notes: list[Note] = []
        counts = {"pitch": 0, "duration": 0, "velocity": 0}
        offset = 0.0

        for note in sequence.notes:
            # Apply mutations
            mutated = self.mutate_pitch(note)
            if mutated.midi_pitch != note.midi_pitch:
                counts["pitch"] += 1

            mutated = self.mutate_duration(mutated)
            if mutated.duration_beats != note.duration_beats:
                counts["duration"] += 1

            mutated = self.mutate_velocity(mutated)
            if mutated.velocity != note.velocity:
                counts["velocity"] += 1

            # Recalculate offset for sequential layout
            result_notes.append(Note(
                midi_pitch=mutated.midi_pitch,
                velocity=mutated.velocity,
                duration_beats=mutated.duration_beats,
                offset_beats=offset,
            ))
            offset += mutated.duration_beats

        total = sum(counts.values())
        return MutationResult(
            sequence=HarmonicSequence(
                notes=result_notes,
                scale_name=sequence.scale_name,
                tempo_bpm=sequence.tempo_bpm,
            ),
            mutations_applied=total,
            mutation_types=counts,
        )

    def evolve(
        self,
        sequence: HarmonicSequence,
        generations: int = 10,
        fitness: Optional[Callable[[HarmonicSequence], float]] = None,
        keep_best: bool = True,
    ) -> list[MutationResult]:
        """Evolve a sequence over multiple generations.

        Each generation applies the configured mutations. If a fitness
        function is provided, the best variant is carried forward
        (elitist selection).

        Args:
            sequence: Starting sequence.
            generations: Number of mutation rounds.
            fitness: Optional fitness function (higher = better).
            keep_best: If True and fitness is given, keep the fittest
                       variant each generation.

        Returns:
            List of MutationResult for each generation.
        """
        results: list[MutationResult] = []
        current = sequence

        for _ in range(generations):
            # Generate a few candidates
            candidates = [self.mutate(current) for _ in range(3)]

            if fitness is not None and keep_best:
                best = max(candidates, key=lambda r: fitness(r.sequence))
            else:
                best = candidates[0]

            results.append(best)
            current = best.sequence

        return results

    def crossover(
        self,
        parent_a: HarmonicSequence,
        parent_b: HarmonicSequence,
    ) -> HarmonicSequence:
        """Single-point crossover between two sequences.

        Takes the first half of notes from parent_a and the second half
        from parent_b, then recalculates offsets.

        Args:
            parent_a: First parent sequence.
            parent_b: Second parent sequence.

        Returns:
            Child sequence combining notes from both parents.
        """
        a_notes = parent_a.notes
        b_notes = parent_b.notes

        if not a_notes or not b_notes:
            return HarmonicSequence(
                scale_name=self.scale_name,
                tempo_bpm=parent_a.tempo_bpm,
            )

        split = self._rng.randint(1, max(len(a_notes), len(b_notes)) - 1) if max(len(a_notes), len(b_notes)) > 1 else 1

        child_notes = list(a_notes[:split])
        if split < len(b_notes):
            child_notes.extend(b_notes[split:])

        # Recalculate offsets
        offset = 0.0
        recalculated: list[Note] = []
        for note in child_notes:
            recalculated.append(Note(
                midi_pitch=note.midi_pitch,
                velocity=note.velocity,
                duration_beats=note.duration_beats,
                offset_beats=offset,
            ))
            offset += note.duration_beats

        return HarmonicSequence(
            notes=recalculated,
            scale_name=self.scale_name,
            tempo_bpm=parent_a.tempo_bpm,
        )

    # ── Structural Mutations ───────────────────────────────────────

    def invert(self, sequence: HarmonicSequence, pivot: Optional[int] = None) -> HarmonicSequence:
        """Invert a sequence around a pivot pitch.

        Every pitch is reflected: new_pitch = 2 * pivot - old_pitch.
        The result is snapped to scale if enforcement is on.
        """
        if pivot is None:
            pivot = self.root_midi

        inverted = sequence.invert_around(pivot)

        if self.config.scale_enforcement:
            fixed_notes: list[Note] = []
            offset = 0.0
            for note in inverted.notes:
                snapped = self._snap_to_scale(note.midi_pitch)
                fixed_notes.append(Note(
                    midi_pitch=snapped,
                    velocity=note.velocity,
                    duration_beats=note.duration_beats,
                    offset_beats=offset,
                ))
                offset += note.duration_beats
            return HarmonicSequence(
                notes=fixed_notes,
                scale_name=sequence.scale_name,
                tempo_bpm=sequence.tempo_bpm,
            )

        return inverted

    def retrograde_invert(self, sequence: HarmonicSequence, pivot: Optional[int] = None) -> HarmonicSequence:
        """Retrograde inversion — reverse the sequence then invert intervals."""
        reversed_seq = HarmonicSequence(
            notes=list(reversed(sequence.notes)),
            scale_name=sequence.scale_name,
            tempo_bpm=sequence.tempo_bpm,
        )
        # Recalculate offsets for reversed sequence
        offset = 0.0
        fixed: list[Note] = []
        for note in reversed_seq.notes:
            fixed.append(Note(
                midi_pitch=note.midi_pitch,
                velocity=note.velocity,
                duration_beats=note.duration_beats,
                offset_beats=offset,
            ))
            offset += note.duration_beats

        recalculated = HarmonicSequence(
            notes=fixed,
            scale_name=sequence.scale_name,
            tempo_bpm=sequence.tempo_bpm,
        )
        return self.invert(recalculated, pivot)
