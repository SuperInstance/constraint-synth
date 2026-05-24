"""Tests for the ratio-based scale system and consonance field."""

import math
from fractions import Fraction
import pytest

from constraint_synth.scales import (
    SCALES, TraditionScale,
    ratio_to_cents, cents_to_ratio, semitones_to_ratio, ratio_to_semitones,
    tenney_height, consonance_score,
    consonance_overlap, find_sangam, tradition_distance,
    PERFECT_FIFTH, MAJOR_THIRD, UNISON,
    KOMAL_GA_SMALL, HIJAZ_AUG_SEC,
    list_traditions, get_scale,
)
from constraint_synth.consonance_field import (
    ConsonanceField, SilencePoint,
)


class TestRatioConversions:
    def test_perfect_fifth_cents(self):
        assert abs(ratio_to_cents(Fraction(3, 2)) - 702.0) < 1.0

    def test_octave_cents(self):
        assert abs(ratio_to_cents(Fraction(2, 1)) - 1200.0) < 0.1

    def test_unison_cents(self):
        assert ratio_to_cents(Fraction(1, 1)) == 0.0

    def test_cents_roundtrip(self):
        """cents → ratio → cents should be approximately the same."""
        for cents in [0, 100, 200, 300, 500, 700, 1200]:
            ratio = cents_to_ratio(cents)
            recovered = ratio_to_cents(ratio)
            assert abs(recovered - cents) < 5.0, f"Cents {cents}: got {recovered}"

    def test_semitones_to_ratio(self):
        # 7 semitones = perfect fifth
        ratio = semitones_to_ratio(7.0)
        assert abs(ratio_to_cents(ratio) - 700.0) < 10.0

    def test_ratio_to_semitones(self):
        assert abs(ratio_to_semitones(Fraction(3, 2)) - 7.0) < 0.1


class TestConsonanceMetrics:
    def test_unison_most_consonant(self):
        assert consonance_score(Fraction(1, 1)) == 1.0

    def test_octave_very_consonant(self):
        assert consonance_score(Fraction(2, 1)) > 0.9

    def test_perfect_fifth_more_than_tritone(self):
        assert consonance_score(Fraction(3, 2)) > consonance_score(Fraction(45, 32))

    def test_tenney_height_ordering(self):
        """Simpler ratios should have lower Tenney height."""
        assert tenney_height(Fraction(3, 2)) < tenney_height(Fraction(45, 32))
        assert tenney_height(Fraction(1, 1)) == 0.0

    def test_consonance_monotonic_with_simplicity(self):
        """More complex ratios should generally have lower consonance."""
        simple = consonance_score(Fraction(3, 2))
        complex_ = consonance_score(Fraction(45, 32))
        assert simple > complex_


class TestTraditionScale:
    def test_all_scales_exist(self):
        assert len(SCALES) >= 26  # 26 defined

    def test_scale_has_intervals(self):
        for name, scale in SCALES.items():
            assert len(scale.intervals) >= 4, f"{name} has only {len(scale.intervals)} intervals"
            for r in scale.intervals:
                assert isinstance(r, Fraction), f"{name} has non-Fraction interval: {r}"

    def test_scale_semitone_approximation(self):
        for name, scale in SCALES.items():
            semitones = scale.semitone_approximation()
            assert len(semitones) == len(scale.intervals)
            for i, s in enumerate(semitones):
                assert -1 < s < 14, f"{name}[{i}]: semitone {s} out of range"

    def test_scale_consonance_profile(self):
        for name, scale in SCALES.items():
            profile = scale.consonance_profile()
            assert len(profile) == len(scale.intervals)
            for c in profile:
                assert 0 <= c <= 1.0

    def test_in_octave_normalizes(self):
        scale = SCALES["major"]
        octave = scale.in_octave()
        for r in octave:
            assert Fraction(1, 1) <= r < Fraction(2, 1)

    def test_at_degree(self):
        scale = SCALES["major"]
        assert scale.at_degree(1) == UNISON
        # 2nd degree should be MAJOR_SECOND
        from constraint_synth.scales import MAJOR_SECOND
        assert scale.at_degree(2) == MAJOR_SECOND

    def test_native_names_present(self):
        """Every scale should have a non-empty native_name."""
        for name, scale in SCALES.items():
            assert scale.native_name, f"{name} missing native_name"

    def test_traditions_represented(self):
        traditions = list_traditions()
        assert "indian" in traditions
        assert "arabic" in traditions
        assert "japanese" in traditions
        assert "western" in traditions
        assert "blues" in traditions or "african_american" in traditions

    def test_get_scale_case_insensitive(self):
        assert get_scale("Bhairavi") is not None
        assert get_scale("BHAIRAVI") is not None

    def test_get_scale_missing(self):
        assert get_scale("nonexistent_scale") is None


class TestConsonanceOverlap:
    def test_bhairavi_hijaz_share_perfect_fifth(self):
        shared = consonance_overlap("bhairavi", "hijaz")
        ratios = [r for r, _, _ in shared]
        # Perfect fifth should be shared
        assert any(abs(ratio_to_cents(r) - 702) < 50 for r in ratios)

    def test_major_minor_share_many(self):
        shared = consonance_overlap("major", "natural_minor")
        assert len(shared) >= 3  # 2nd, 4th, 5th at minimum

    def test_self_overlap(self):
        """A scale should overlap with itself completely."""
        shared = consonance_overlap("major", "major")
        assert len(shared) >= len(SCALES["major"].intervals)


class TestSangam:
    def test_universal_sangam_has_fifth(self):
        """The perfect fifth should appear in any universal Sangam."""
        result = find_sangam(["major", "bhairavi", "hijaz", "hirajoshi", "blues"])
        assert len(result) >= 1
        # Perfect fifth should be there
        assert any(abs(ratio_to_cents(r) - 702) < 50 for r in result)

    def test_single_tradition_returns_empty(self):
        result = find_sangam(["major"])
        assert result == []

    def test_identical_scales_full_sangam(self):
        result = find_sangam(["major", "major"])
        assert len(result) >= 5  # most intervals should match


class TestTraditionDistance:
    def test_self_distance_is_zero(self):
        assert tradition_distance("major", "major") == 0.0

    def test_similar_scales_close(self):
        """Major and Mixolydian should be close (only differ by 7th)."""
        d = tradition_distance("major", "mixolydian")
        assert d < 30.0

    def test_distant_scales_far(self):
        """Major and whole tone should be further apart."""
        d1 = tradition_distance("major", "mixolydian")
        d2 = tradition_distance("major", "whole_tone")
        assert d2 > d1


class TestSilencePoint:
    def test_tension_at_center(self):
        sp = SilencePoint(Fraction(1, 1), 2.0, 0.8)
        assert sp.tension_at(0) == 0.8

    def test_tension_decays_with_distance(self):
        sp = SilencePoint(Fraction(1, 1), 2.0, 0.8)
        near = sp.tension_at(10)
        far = sp.tension_at(500)
        assert near > far

    def test_decay_modes(self):
        for mode in ["resonant", "abrupt", "breathing"]:
            sp = SilencePoint(Fraction(1, 1), 1.0, 0.5, mode)
            assert sp.decay_mode == mode


class TestConsonanceField:
    def test_basic_consonance(self):
        field = ConsonanceField()
        # Unison should be most consonant
        c = field.consonance_at(Fraction(1, 1))
        assert c > 0.9

    def test_perfect_fifth_more_consonant_than_tritone(self):
        field = ConsonanceField()
        assert field.consonance_at(Fraction(3, 2)) > field.consonance_at(Fraction(45, 32))

    def test_gradient_is_defined(self):
        """Gradient should be finite and defined for any ratio."""
        field = ConsonanceField()
        for ratio in [Fraction(45, 32), Fraction(3, 2), Fraction(1, 1)]:
            up, down = field.gradient_at(ratio)
            assert math.isfinite(up) and math.isfinite(down)

    def test_find_nearest_peak(self):
        field = ConsonanceField()
        # From tritone, nearest peak should be perfect fifth or fourth
        peak = field.find_nearest_peak(Fraction(45, 32))
        # The peak should be more consonant than where we started
        assert field.consonance_at(peak) > field.consonance_at(Fraction(45, 32))

    def test_tension_profile(self):
        field = ConsonanceField()
        intervals = [Fraction(9, 8), Fraction(5, 4), Fraction(45, 32), Fraction(3, 2)]
        tensions = field.tension_profile(intervals)
        assert len(tensions) == 4
        # Tritone (index 2) should have highest tension
        assert tensions[2] == max(tensions)

    def test_find_silence_positions(self):
        field = ConsonanceField()
        # Chromatic ascent — tension should peak somewhere
        chromatic = [Fraction(2) ** Fraction(i, 12) for i in range(1, 13)]
        positions = field.find_silence_positions(chromatic)
        # Should find at least one position
        assert len(positions) >= 1

    def test_silence_modifies_field(self):
        field = ConsonanceField()
        before = field.consonance_at(Fraction(3, 2))
        field.add_silence(SilencePoint(Fraction(3, 2), 2.0, 0.8))
        after = field.consonance_at(Fraction(3, 2))
        assert after > before  # silence point should increase consonance nearby

    def test_shared_consonance_between_traditions(self):
        field = ConsonanceField()
        shared = field.find_shared_consonance(
            SCALES["bhairavi"].intervals,
            SCALES["hijaz"].intervals,
        )
        assert len(shared) >= 1
        # Perfect fifth should be in there
        assert any(abs(ratio_to_cents(r) - 702) < 50 for r, _ in shared)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
