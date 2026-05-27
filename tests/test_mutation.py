"""Tests for constraint_synth.mutation — ConstraintMutation."""

import pytest

from constraint_synth.harmony import HarmonicSequence, HarmonicSynthesizer, Note
from constraint_synth.mutation import (
    MutationConfig,
    MutationResult,
    ConstraintMutation,
)


def _make_sequence(pitches: list[int], dur: float = 1.0) -> HarmonicSequence:
    """Helper: create a simple sequence from pitch list."""
    notes = []
    offset = 0.0
    for p in pitches:
        notes.append(Note(midi_pitch=p, duration_beats=dur, offset_beats=offset))
        offset += dur
    return HarmonicSequence(notes=notes, scale_name="major", tempo_bpm=120.0)


# ── MutationConfig ──────────────────────────────────────────────────

class TestMutationConfig:
    def test_defaults(self):
        cfg = MutationConfig()
        assert cfg.pitch_mutation_rate == 0.3
        assert cfg.duration_mutation_rate == 0.2
        assert cfg.velocity_mutation_rate == 0.1
        assert cfg.max_pitch_shift == 4
        assert cfg.preserve_root is True
        assert cfg.scale_enforcement is True

    def test_custom_config(self):
        cfg = MutationConfig(
            pitch_mutation_rate=0.5,
            max_pitch_shift=7,
            preserve_root=False,
        )
        assert cfg.pitch_mutation_rate == 0.5
        assert cfg.max_pitch_shift == 7
        assert cfg.preserve_root is False


# ── ConstraintMutation Init ─────────────────────────────────────────

class TestConstraintMutationInit:
    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            ConstraintMutation(scale_name="nonexistent")

    def test_valid_init(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60)
        assert cm.scale_name == "major"

    def test_scale_pitches_built(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60)
        assert 60 in cm._scale_pitches
        assert 62 in cm._scale_pitches
        assert 64 in cm._scale_pitches


# ── Snap to Scale ───────────────────────────────────────────────────

class TestSnapToScale:
    def test_exact_match(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60)
        # 64 is E in C major
        assert cm._snap_to_scale(64) == 64

    def test_snap_to_nearest(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60)
        # 63 (C#) should snap to 62 (D) or 64 (E)
        snapped = cm._snap_to_scale(63)
        assert snapped in (62, 64)

    def test_out_of_range_clamps(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, config=MutationConfig())
        # Pitch 1 should snap to something valid
        snapped = cm._snap_to_scale(1)
        assert snapped in cm._scale_pitches


# ── Single-Note Mutations ──────────────────────────────────────────

class TestSingleNoteMutations:
    def test_pitch_mutation_with_100_percent_rate(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(pitch_mutation_rate=1.0, max_pitch_shift=2),
        )
        note = Note(midi_pitch=64)  # E — not root
        mutated = cm.mutate_pitch(note)
        # Should have changed (rate=100%) but stayed on scale
        assert mutated.midi_pitch in cm._scale_pitches

    def test_pitch_mutation_preserves_root(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(pitch_mutation_rate=1.0, preserve_root=True),
        )
        note = Note(midi_pitch=60)  # Root
        mutated = cm.mutate_pitch(note)
        assert mutated.midi_pitch == 60

    def test_pitch_mutation_can_change_root_when_disabled(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(pitch_mutation_rate=1.0, preserve_root=False, max_pitch_shift=2),
        )
        note = Note(midi_pitch=60)
        mutated = cm.mutate_pitch(note)
        # May or may not change, but it's allowed to
        assert mutated.midi_pitch in cm._scale_pitches

    def test_zero_rate_no_mutation(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60,
            config=MutationConfig(pitch_mutation_rate=0.0),
        )
        note = Note(midi_pitch=64, duration_beats=1.0, velocity=100)
        mutated = cm.mutate_pitch(note)
        assert mutated.midi_pitch == 64

    def test_duration_mutation(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(duration_mutation_rate=1.0),
        )
        note = Note(midi_pitch=64, duration_beats=1.0)
        mutated = cm.mutate_duration(note)
        assert mutated.duration_beats in cm.config.duration_options

    def test_velocity_mutation(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(velocity_mutation_rate=1.0),
        )
        note = Note(midi_pitch=64, velocity=100)
        mutated = cm.mutate_velocity(note)
        lo, hi = cm.config.velocity_range
        assert lo <= mutated.velocity <= hi


# ── Sequence Mutation ───────────────────────────────────────────────

class TestMutate:
    def test_mutate_preserves_note_count(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=42)
        seq = _make_sequence([60, 62, 64, 65, 67])
        result = cm.mutate(seq)
        assert len(result.sequence.notes) == len(seq.notes)

    def test_mutate_returns_stats(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(
                pitch_mutation_rate=1.0,
                duration_mutation_rate=1.0,
                velocity_mutation_rate=1.0,
                preserve_root=False,
            ),
        )
        seq = _make_sequence([62, 64, 67])
        result = cm.mutate(seq)
        assert isinstance(result, MutationResult)
        assert isinstance(result.mutation_types, dict)
        assert "pitch" in result.mutation_types
        assert "duration" in result.mutation_types
        assert "velocity" in result.mutation_types

    def test_mutated_pitches_stay_on_scale(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(
                pitch_mutation_rate=1.0,
                preserve_root=False,
                scale_enforcement=True,
            ),
        )
        seq = _make_sequence([60, 64, 67])
        for _ in range(10):
            result = cm.mutate(seq)
            for note in result.sequence.notes:
                assert note.midi_pitch in cm._scale_pitches

    def test_mutate_sequential_offsets(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
        )
        seq = _make_sequence([60, 62, 64, 67])
        result = cm.mutate(seq)
        expected_offset = 0.0
        for note in result.sequence.notes:
            assert note.offset_beats == pytest.approx(expected_offset, abs=1e-6)
            expected_offset += note.duration_beats


# ── Evolution ───────────────────────────────────────────────────────

class TestEvolve:
    def test_evolve_generations(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=42)
        seq = _make_sequence([60, 62, 64, 67])
        results = cm.evolve(seq, generations=5)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, MutationResult)

    def test_evolve_with_fitness(self):
        def fitness(s: HarmonicSequence) -> float:
            # Prefer higher average pitch
            if not s.notes:
                return 0.0
            return sum(n.midi_pitch for n in s.notes) / len(s.notes)

        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(pitch_mutation_rate=0.8, preserve_root=False),
        )
        seq = _make_sequence([60, 62, 64])
        results = cm.evolve(seq, generations=5, fitness=fitness, keep_best=True)
        assert len(results) == 5
        # Each generation should produce a result
        for r in results:
            assert len(r.sequence.notes) > 0


# ── Crossover ───────────────────────────────────────────────────────

class TestCrossover:
    def test_crossover_combines_parents(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=42)
        parent_a = _make_sequence([60, 62, 64, 67])
        parent_b = _make_sequence([69, 71, 72, 74])
        child = cm.crossover(parent_a, parent_b)
        assert len(child.notes) > 0
        assert child.scale_name == cm.scale_name

    def test_crossover_sequential_offsets(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=42)
        parent_a = _make_sequence([60, 62, 64])
        parent_b = _make_sequence([67, 69, 71])
        child = cm.crossover(parent_a, parent_b)
        offset = 0.0
        for note in child.notes:
            assert note.offset_beats == pytest.approx(offset, abs=1e-6)
            offset += note.duration_beats

    def test_crossover_empty_parent(self):
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=42)
        parent_a = _make_sequence([60, 62])
        parent_b = HarmonicSequence()
        child = cm.crossover(parent_a, parent_b)
        assert len(child.notes) == 0  # empty parent_b → empty child


# ── Structural Mutations ────────────────────────────────────────────

class TestStructuralMutations:
    def test_invert(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(scale_enforcement=True),
        )
        seq = _make_sequence([60, 64, 67])
        inverted = cm.invert(seq, pivot=64)
        # All inverted pitches should be on scale
        for note in inverted.notes:
            assert note.midi_pitch in cm._scale_pitches

    def test_retrograde_invert(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(scale_enforcement=True),
        )
        seq = _make_sequence([60, 62, 64])
        ri = cm.retrograde_invert(seq, pivot=62)
        for note in ri.notes:
            assert note.midi_pitch in cm._scale_pitches

    def test_invert_without_scale_enforcement(self):
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(scale_enforcement=False),
        )
        seq = _make_sequence([60, 64, 67])
        inverted = cm.invert(seq, pivot=64)
        # Without scale enforcement, pitches are raw inversions
        # 2*64 - 60 = 68, 2*64 - 64 = 64, 2*64 - 67 = 61
        assert inverted.notes[1].midi_pitch == 64  # pivot stays


# ── Integration with HarmonicSynthesizer ────────────────────────────

class TestIntegration:
    def test_mutate_generated_melody(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        cm = ConstraintMutation(scale_name="major", root_midi=60, rng_seed=99)
        melody = hs.generate_melody(num_notes=16)
        result = cm.mutate(melody)
        assert len(result.sequence.notes) == len(melody.notes)
        for note in result.sequence.notes:
            assert note.midi_pitch in cm._scale_pitches

    def test_evolve_chord_progression(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        cm = ConstraintMutation(
            scale_name="major", root_midi=60, rng_seed=42,
            config=MutationConfig(pitch_mutation_rate=0.5, preserve_root=False),
        )
        prog = hs.generate_chord_progression(degrees=[1, 4, 5, 1])
        results = cm.evolve(prog, generations=3)
        assert len(results) == 3
        for r in results:
            for note in r.sequence.notes:
                assert note.midi_pitch in cm._scale_pitches

    def test_cross_cultural_mutation(self):
        """Ensure mutations work across all cultural scales."""
        for scale in ["bhairavi", "hijaz", "hirajoshi", "blues", "dorian"]:
            hs = HarmonicSynthesizer(scale_name=scale, root_midi=60, rng_seed=42)
            cm = ConstraintMutation(scale_name=scale, root_midi=60, rng_seed=42)
            melody = hs.generate_melody(num_notes=8)
            result = cm.mutate(melody)
            for note in result.sequence.notes:
                assert note.midi_pitch in cm._scale_pitches, (
                    f"Scale {scale}: pitch {note.midi_pitch} not in scale"
                )
