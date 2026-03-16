"""Tests for ReactionLibrary caching methods."""

import pytest

from rxn_ca.reactions import ReactionLibrary, ScoredReactionSet
from rxn_ca.phases import SolidPhaseSet


@pytest.fixture
def simple_phase_set():
    """Create a minimal phase set for testing."""
    return SolidPhaseSet(
        phases=["BaO", "TiO2", "BaTiO3"],
        volumes={"BaO": 25.0, "TiO2": 18.8, "BaTiO3": 38.0},
        melting_points={"BaO": 2196, "TiO2": 2116, "BaTiO3": 1898},
        densities={"BaO": 5.72, "TiO2": 4.23, "BaTiO3": 6.02},
        experimentally_observed={"BaO": True, "TiO2": True, "BaTiO3": True},
    )


@pytest.fixture
def empty_library(simple_phase_set):
    """Create an empty ReactionLibrary."""
    return ReactionLibrary(phases=simple_phase_set)


@pytest.fixture
def library_with_temps(simple_phase_set):
    """Create a ReactionLibrary with some temperatures."""
    lib = ReactionLibrary(phases=simple_phase_set)
    # Add empty reaction sets at several temps
    lib.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 300)
    lib.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 400)
    lib.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 500)
    return lib


class TestReactionLibraryCaching:
    """Tests for caching-related methods on ReactionLibrary."""

    def test_has_temp_empty(self, empty_library):
        """Empty library has no temps."""
        assert empty_library.has_temp(300) is False
        assert empty_library.has_temp(400) is False

    def test_has_temp_with_data(self, library_with_temps):
        """Library reports temps it contains."""
        assert library_with_temps.has_temp(300) is True
        assert library_with_temps.has_temp(400) is True
        assert library_with_temps.has_temp(500) is True
        assert library_with_temps.has_temp(600) is False

    def test_has_temp_converts_to_int(self, library_with_temps):
        """has_temp converts to int for comparison."""
        assert library_with_temps.has_temp(300.0) is True
        assert library_with_temps.has_temp(300) is True

    def test_get_missing_temps_all_missing(self, empty_library):
        """All temps missing from empty library."""
        temps = [300, 400, 500]
        missing = empty_library.get_missing_temps(temps)

        assert missing == [300, 400, 500]

    def test_get_missing_temps_none_missing(self, library_with_temps):
        """No temps missing when all present."""
        temps = [300, 400, 500]
        missing = library_with_temps.get_missing_temps(temps)

        assert missing == []

    def test_get_missing_temps_partial(self, library_with_temps):
        """Only returns temps not in library."""
        temps = [300, 400, 600, 700]
        missing = library_with_temps.get_missing_temps(temps)

        assert 300 not in missing
        assert 400 not in missing
        assert 600 in missing
        assert 700 in missing
        assert len(missing) == 2

    def test_get_missing_temps_converts_to_int(self, library_with_temps):
        """get_missing_temps converts to int."""
        temps = [300.0, 600.0]
        missing = library_with_temps.get_missing_temps(temps)

        assert missing == [600]
        assert isinstance(missing[0], int)

    def test_merge_into_empty(self, empty_library, library_with_temps):
        """Can merge into empty library."""
        empty_library.merge(library_with_temps)

        assert empty_library.has_temp(300)
        assert empty_library.has_temp(400)
        assert empty_library.has_temp(500)

    def test_merge_adds_new_temps(self, library_with_temps, simple_phase_set):
        """Merge adds temps not already present."""
        other = ReactionLibrary(phases=simple_phase_set)
        other.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 600)
        other.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 700)

        library_with_temps.merge(other)

        assert library_with_temps.has_temp(300)  # Original
        assert library_with_temps.has_temp(600)  # Added
        assert library_with_temps.has_temp(700)  # Added

    def test_merge_doesnt_overwrite(self, simple_phase_set):
        """Merge doesn't overwrite existing temps."""
        lib1 = ReactionLibrary(phases=simple_phase_set)
        lib1.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 300)

        lib2 = ReactionLibrary(phases=simple_phase_set)
        lib2.add_rxns_at_temp(ScoredReactionSet([], simple_phase_set), 300)

        original_rxn_set = lib1.get_rxns_at_temp(300)
        lib1.merge(lib2)

        # Should still have original reaction set
        assert lib1.get_rxns_at_temp(300) is original_rxn_set

    def test_merge_returns_self(self, empty_library, library_with_temps):
        """Merge returns self for chaining."""
        result = empty_library.merge(library_with_temps)

        assert result is empty_library

    def test_temps_property(self, library_with_temps):
        """temps property returns list of temperatures."""
        temps = library_with_temps.temps

        assert isinstance(temps, list)
        assert 300 in temps
        assert 400 in temps
        assert 500 in temps
        assert len(temps) == 3


class TestReactionLibraryIntegration:
    """Integration tests using real serialized data."""

    @pytest.mark.skip(reason="Test data file has outdated format (missing densities)")
    def test_from_file_has_temps(self, get_test_file_path):
        """Library loaded from file reports correct temps."""
        lib = ReactionLibrary.from_file(
            get_test_file_path("integration/batio3_library.json")
        )

        # The test file should have at least one temperature
        assert len(lib.temps) > 0

        # has_temp should work
        first_temp = lib.temps[0]
        assert lib.has_temp(first_temp) is True

    @pytest.mark.skip(reason="Test data file has outdated format (missing densities)")
    def test_get_missing_temps_with_real_data(self, get_test_file_path):
        """get_missing_temps works with real data."""
        lib = ReactionLibrary.from_file(
            get_test_file_path("integration/batio3_library.json")
        )

        existing_temp = lib.temps[0]
        nonexistent_temp = 99999

        missing = lib.get_missing_temps([existing_temp, nonexistent_temp])

        assert existing_temp not in missing
        assert nonexistent_temp in missing
