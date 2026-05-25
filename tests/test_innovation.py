"""Tests for innovation_cycle, perception, lattice, and neural modules."""

import pytest
import math
from fractions import Fraction

from constraint_synth.innovation_cycle import (
    Phase,
    Style,
    WESTERN_STYLES,
    detect_phase,
    cycle_acceleration,
    predict_next_rebellion,
)
from constraint_synth.perception import (
    jnd,
    consonance_threshold,
    tradition_recognition,
    pleasantness,
    MOST_PLEASING,
)
from constraint_synth.lattice import (
    EisensteinNorm,
    LatticePoint,
    tenney_height,
    consonance_score,
    find_sangam,
    NEAREST_HARMONIC,
)
from constraint_synth.neural import (
    predict_fmr,
    predict_eeg,
    adaptation_rate,
    DIAL_BRAIN_CORRELATION,
)


# ── Innovation Cycle Tests ───────────────────────────────────────────────

class TestInnovationCycle:
    def test_phase_enum(self):
        assert Phase.DISCOVERY.value == "discovery"
        assert len(Phase) == 5

    def test_western_styles_exist(self):
        assert len(WESTERN_STYLES) >= 10

    def test_style_attributes(self):
        bebop = [s for s in WESTERN_STYLES if s.name == "Bebop"][0]
        assert bebop.year_start == 1942
        assert len(bebop.dial_position) == 3

    def test_cycle_acceleration_returns_data(self):
        data = cycle_acceleration()
        assert len(data) >= 10
        # Each entry is (name, year, cycle_time)
        assert all(isinstance(t[1], int) for t in data)
        assert all(isinstance(t[2], int) for t in data)

    def test_cycle_times_decreasing_trend(self):
        """Cycle times should generally decrease over time."""
        data = cycle_acceleration()
        times = [t[2] for t in data]
        # First should be larger than last (general trend)
        assert times[0] > times[-1]

    def test_predict_next_rebellion(self):
        result = predict_next_rebellion(2026)
        assert result["current_year"] == 2026
        assert result["predicted_cycle_years"] > 0
        assert result["predicted_cycle_years"] <= 200
        assert result["predicted_rebellion_year"] > 2026

    def test_detect_phase_boredom(self):
        style = WESTERN_STYLES[0]  # Renaissance, definitely bored
        phase = detect_phase(style, metrics={"in_school": True})
        assert phase == Phase.BOREDOM

    def test_detect_phase_ubiquity(self):
        style = WESTERN_STYLES[0]
        phase = detect_phase(style, metrics={"ubiquitous": True})
        assert phase == Phase.UBIQUITY

    def test_detect_phase_codification(self):
        style = WESTERN_STYLES[0]
        phase = detect_phase(style, metrics={"codified": True})
        assert phase == Phase.CODIFICATION


# ── Perception Tests ─────────────────────────────────────────────────────

class TestPerception:
    def test_jnd_values(self):
        """JND values should match experimental findings."""
        assert jnd("I_vert") == 0.15
        assert jnd("I_horiz") == 0.21
        assert jnd("I_spectral") == 0.35

    def test_jnd_vert_most_sensitive(self):
        """I_vert (harmony) should have smallest JND (most perceptible)."""
        assert jnd("I_vert") < jnd("I_horiz") < jnd("I_spectral")

    def test_jnd_invalid_axis(self):
        with pytest.raises(ValueError):
            jnd("I_invalid")

    def test_consonance_threshold_unison(self):
        """Unison (0 cents) should be maximally consonant."""
        assert consonance_threshold(0) > 0.9

    def test_consonance_threshold_octave(self):
        """Octave (1200 cents) should be very consonant."""
        assert consonance_threshold(1200) > 0.8

    def test_consonance_threshold_tritone(self):
        """Tritone (~600 cents) should be less consonant."""
        assert consonance_threshold(600) < consonance_threshold(700)

    def test_consonance_threshold_fifth(self):
        """Perfect fifth (700 cents) should be more consonant than tritone."""
        assert consonance_threshold(700) > consonance_threshold(600)

    def test_tradition_recognition_accuracy(self):
        """Recognition should correctly identify all 10 traditions."""
        from constraint_synth.dial_space import TRADITIONS
        correct = 0
        total = 0
        for name, pos in TRADITIONS.items():
            predicted, confidence = tradition_recognition(pos.as_tuple())
            if predicted == name:
                correct += 1
            total += 1
            assert confidence > 0.5, f"{name}: confidence {confidence} too low"

        accuracy = correct / total
        assert accuracy >= 0.95, f"Recognition accuracy {accuracy:.1%} < 95%"

    def test_most_pleasing_position(self):
        assert len(MOST_PLEASING) == 3
        # The most pleasing position should have high pleasantness
        p = pleasantness(MOST_PLEASING)
        assert p > 0.5

    def test_pleasantness_range(self):
        """Pleasantness should be in [0, 1]."""
        for pos in [(1.0, 1.0, 1.0), (4.0, 4.0, 4.0), (2.5, 2.5, 2.5)]:
            p = pleasantness(pos)
            assert 0.0 <= p <= 1.0

    def test_pleasantness_favors_moderate_complexity(self):
        """Moderate I_vert should score higher than extremes."""
        moderate = pleasantness((2.5, 2.5, 2.5))
        extreme_high = pleasantness((4.0, 4.0, 4.0))
        assert moderate > extreme_high


# ── Lattice Tests ────────────────────────────────────────────────────────

class TestLattice:
    def test_lattice_point_unison(self):
        p = LatticePoint(0, 0, 0)
        assert p.ratio == Fraction(1, 1)
        assert p.tenney_height() == 0.0
        assert p.consonance() == 1.0

    def test_lattice_point_octave(self):
        p = LatticePoint(1, 0, 0)
        assert p.ratio == Fraction(2, 1)

    def test_lattice_point_fifth(self):
        p = LatticePoint(0, 1, 0)
        assert p.ratio == Fraction(3, 1)
        assert p.normalized_cents() == pytest.approx(702.0, abs=5.0)

    def test_lattice_point_major_third(self):
        p = LatticePoint(0, 0, 1)
        assert p.ratio == Fraction(5, 1)

    def test_tenney_height_simple(self):
        assert tenney_height(Fraction(1, 1)) == 0.0
        assert tenney_height(Fraction(2, 1)) == pytest.approx(1.0)

    def test_tenney_height_ordering(self):
        """Simpler ratios should have lower Tenney height."""
        th_unison = tenney_height(Fraction(1, 1))
        th_octave = tenney_height(Fraction(2, 1))
        th_fifth = tenney_height(Fraction(3, 2))
        th_complex = tenney_height(Fraction(15, 8))
        assert th_unison < th_octave < th_fifth < th_complex

    def test_consonance_score_frequencies(self):
        """Unison should be maximally consonant, fifth more than tritone."""
        assert consonance_score(440.0, 440.0) > 0.9
        assert consonance_score(440.0, 660.0) > consonance_score(440.0, 622.25)

    def test_eisenstein_norm(self):
        n = EisensteinNorm.norm(Fraction(1, 1))
        assert n == 0.0

    def test_nearest_harmonic_built(self):
        assert len(NEAREST_HARMONIC) > 0
        # 700 cents should map to something near 3/2
        frac_700, th_700 = NEAREST_HARMONIC[700]
        cents = 1200.0 * math.log2(float(frac_700))
        assert abs(cents - 700) < 10

    def test_find_sangam(self):
        """Test sangam finding with simple scale data."""
        traditions = {
            "trad_a": [Fraction(1, 1), Fraction(3, 2), Fraction(5, 4)],
            "trad_b": [Fraction(1, 1), Fraction(3, 2), Fraction(4, 3)],
            "trad_c": [Fraction(1, 1), Fraction(5, 4), Fraction(4, 3)],
            "trad_d": [Fraction(1, 1), Fraction(3, 2)],
        }
        sangam = find_sangam(traditions)
        # 3/2 should be in sangam (appears in 3+ traditions)
        sangam_floats = [float(s) for s in sangam]
        # 1.5 (3/2) should be present
        assert any(abs(s - 1.5) < 0.01 for s in sangam_floats)


# ── Neural Tests ─────────────────────────────────────────────────────────

class TestNeural:
    def test_predict_fmr_returns_regions(self):
        result = predict_fmr((2.5, 2.5, 2.5))
        assert "orbitofrontal_cortex" in result
        assert "nucleus_accumbens" in result
        assert "amygdala" in result
        # All values should be in [0, 1]
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_predict_fmr_consonant_vs_dissonant(self):
        """Consonant position should activate reward regions more."""
        consonant = predict_fmr((1.5, 1.5, 1.5))
        dissonant = predict_fmr((3.5, 3.5, 3.5))
        assert consonant["orbitofrontal_cortex"] > dissonant["orbitofrontal_cortex"]
        assert consonant["nucleus_accumbens"] > dissonant["nucleus_accumbens"]

    def test_predict_eeg_returns_keys(self):
        result = predict_eeg((2.5, 2.5, 2.5))
        assert "brainstem_FFR_amplitude" in result
        assert "consonance_score" in result
        assert 0.0 <= result["brainstem_FFR_amplitude"] <= 1.0

    def test_predict_eeg_consonant_higher_ffr(self):
        """Consonant position should have higher FFR amplitude."""
        cons = predict_eeg((1.5, 1.5, 1.5))
        diss = predict_eeg((3.5, 3.5, 3.5))
        assert cons["brainstem_FFR_amplitude"] > diss["brainstem_FFR_amplitude"]

    def test_adaptation_rate_ordering(self):
        """Boredom should have fastest adaptation (smallest half-life)."""
        assert adaptation_rate("boredom") < adaptation_rate("ubiquity")
        assert adaptation_rate("ubiquity") < adaptation_rate("discovery")
        assert adaptation_rate("rebellion") > adaptation_rate("boredom")

    def test_adaptation_rate_invalid(self):
        with pytest.raises(ValueError):
            adaptation_rate("invalid_phase")

    def test_dial_brain_correlation(self):
        """The correlation should be a reasonable value."""
        assert 0.8 <= DIAL_BRAIN_CORRELATION <= 0.95
