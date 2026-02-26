"""Tests for the get_entries utility function."""

import pytest

from rxn_ca.utilities.get_entries import get_entries


@pytest.mark.integration
def test_get_entries_basic():
    """Test basic get_entries functionality."""
    entries = get_entries(
        chem_sys="Li-O",
        metastability_cutoff=0.1,
        exclude_theoretical_phases=False,
    )

    formulas = {e.composition.reduced_formula for e in entries}
    assert "Li2O" in formulas
    assert "O2" in formulas


@pytest.mark.integration
def test_get_entries_ensure_phases_overrides_theoretical_filter():
    """Test that ensure_phases keeps phases even if marked theoretical in MP.

    This is a regression test for a bug where ensure_phases would not
    actually ensure inclusion of phases that were filtered out by
    exclude_theoretical_phases=True.

    Ca3Co4O9 is a real experimentally-synthesized thermoelectric material,
    but it's marked as 'theoretical' in Materials Project. When we have
    experimental evidence that a phase exists (e.g., from synthesis records),
    ensure_phases should override the theoretical filter.
    """
    # Ca3Co4O9 is marked as theoretical in MP but is experimentally real
    ensure_phases = ["Ca3Co4O9", "CaCO3", "Co3O4"]

    entries = get_entries(
        chem_sys="C-Ca-Co-O",
        metastability_cutoff=0.1,
        ensure_phases=ensure_phases,
        exclude_theoretical_phases=True,
    )

    formulas = {e.composition.reduced_formula for e in entries}

    # All ensure_phases should be present, even the "theoretical" one
    for phase in ensure_phases:
        assert phase in formulas, (
            f"{phase} should be included via ensure_phases "
            f"even if marked theoretical in MP"
        )


@pytest.mark.integration
def test_get_entries_ensure_phases_without_theoretical_filter():
    """Test ensure_phases works when theoretical filter is off."""
    ensure_phases = ["Ca3Co4O9"]

    entries = get_entries(
        chem_sys="C-Ca-Co-O",
        metastability_cutoff=0.1,
        ensure_phases=ensure_phases,
        exclude_theoretical_phases=False,
    )

    formulas = {e.composition.reduced_formula for e in entries}
    assert "Ca3Co4O9" in formulas


@pytest.mark.integration
def test_get_entries_theoretical_phases_excluded_by_default():
    """Test that theoretical phases are excluded when not in ensure_phases."""
    # Don't include Ca3Co4O9 in ensure_phases
    entries = get_entries(
        chem_sys="C-Ca-Co-O",
        metastability_cutoff=0.1,
        ensure_phases=[],
        exclude_theoretical_phases=True,
    )

    formulas = {e.composition.reduced_formula for e in entries}

    # Ca3Co4O9 should NOT be present since it's theoretical and not ensured
    assert "Ca3Co4O9" not in formulas, (
        "Ca3Co4O9 should be excluded as theoretical when not in ensure_phases"
    )
