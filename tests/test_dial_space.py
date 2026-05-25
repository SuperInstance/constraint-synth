"""Tests for dial_space module."""

import pytest
import math
from constraint_synth.dial_space import (
    DialPosition,
    TRADITIONS,
    find_cluster,
    find_nearest_tradition,
    find_unexplored,
    interpolate_traditions,
    structure_surplus,
)


class TestDialPosition:
    def test_distance_to_self(self):
        p = DialPosition(2.5, 2.5, 2.5)
        assert p.distance_to(p) == 0.0

    def test_distance_symmetry(self):
        a = DialPosition(1.0, 2.0, 3.0)
        b = DialPosition(3.0, 2.0, 1.0)
        assert abs(a.distance_to(b) - b.distance_to(a)) < 1e-10

    def test_known_distance(self):
        a = DialPosition(0.0, 0.0, 0.0)
        b = DialPosition(3.0, 4.0, 0.0)
        assert abs(a.distance_to(b) - 5.0) < 1e-10

    def test_as_tuple(self):
        p = DialPosition(1.5, 2.5, 3.5)
        assert p.as_tuple() == (1.5, 2.5, 3.5)


class TestTraditions:
    def test_all_ten_traditions(self):
        assert len(TRADITIONS) == 10

    def test_tradition_names(self):
        expected = {
            "Hindustani", "Carnatic", "Arabic", "Turkish", "Javanese",
            "Balinese", "Gagaku", "Chinese", "West African", "Western ET",
        }
        assert set(TRADITIONS.keys()) == expected

    def test_all_positions_in_range(self):
        for name, pos in TRADITIONS.items():
            assert 1.0 <= pos.I_vert <= 4.0, f"{name} I_vert out of range"
            assert 1.0 <= pos.I_horiz <= 4.0, f"{name} I_horiz out of range"
            assert 1.0 <= pos.I_spectral <= 4.0, f"{name} I_spectral out of range"


class TestFindCluster:
    def test_exact_match(self):
        western = TRADITIONS["Western ET"]
        cluster = find_cluster(western)
        assert cluster[0][0] == "Western ET"
        assert cluster[0][1] < 0.01

    def test_nearby_traditions(self):
        # Hindustani and Carnatic are close (same I_vert)
        cluster = find_cluster(TRADITIONS["Hindustani"])
        names_in_top3 = [name for name, _ in cluster[:3]]
        assert "Carnatic" in names_in_top3

    def test_returns_all_traditions(self):
        cluster = find_cluster(DialPosition(2.5, 2.5, 2.5))
        assert len(cluster) == 10


class TestFindNearestTradition:
    def test_western_position(self):
        western = TRADITIONS["Western ET"]
        name, dist = find_nearest_tradition(western)
        assert name == "Western ET"
        assert dist < 0.01

    def test_gagaku_is_isolated(self):
        # Gagaku has very low I_horiz (1.70), should be nearest to itself
        gagaku = TRADITIONS["Gagaku"]
        name, _ = find_nearest_tradition(gagaku)
        assert name == "Gagaku"


class TestInterpolation:
    def test_endpoint_a(self):
        a = DialPosition(1.0, 1.0, 1.0)
        b = DialPosition(3.0, 3.0, 3.0)
        result = interpolate_traditions(a, b, 0.0)
        assert abs(result.I_vert - 1.0) < 1e-10
        assert abs(result.I_horiz - 1.0) < 1e-10
        assert abs(result.I_spectral - 1.0) < 1e-10

    def test_endpoint_b(self):
        a = DialPosition(1.0, 1.0, 1.0)
        b = DialPosition(3.0, 3.0, 3.0)
        result = interpolate_traditions(a, b, 1.0)
        assert abs(result.I_vert - 3.0) < 1e-10

    def test_midpoint(self):
        a = DialPosition(1.0, 1.0, 1.0)
        b = DialPosition(3.0, 3.0, 3.0)
        result = interpolate_traditions(a, b, 0.5)
        assert abs(result.I_vert - 2.0) < 1e-10

    def test_between_traditions(self):
        h = TRADITIONS["Hindustani"]
        a = TRADITIONS["Arabic"]
        mid = interpolate_traditions(h, a, 0.5)
        # Midpoint should be between the two
        assert h.I_vert <= mid.I_vert <= a.I_vert or a.I_vert <= mid.I_vert <= h.I_vert


class TestFindUnexplored:
    def test_returns_positions(self):
        results = find_unexplored(n_grid=5)
        assert len(results) > 0
        assert all(isinstance(p, DialPosition) for p in results)

    def test_unexplored_are_far_from_traditions(self):
        results = find_unexplored(n_grid=5)
        # All results should be at least 0.5 from any tradition
        for pos in results:
            min_dist = min(pos.distance_to(t) for t in TRADITIONS.values())
            assert min_dist > 0.5


class TestStructureSurplus:
    def test_at_tradition_is_positive(self):
        western = TRADITIONS["Western ET"]
        s = structure_surplus(western)
        assert s > 0

    def test_between_traditions_is_positive(self):
        mid = interpolate_traditions(
            TRADITIONS["Hindustani"],
            TRADITIONS["Arabic"],
            0.5,
        )
        s = structure_surplus(mid)
        assert s > 0
