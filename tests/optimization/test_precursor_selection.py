"""Tests for precursor selection and generation."""

import pytest
from rxn_ca.optimization.precursor_selection import (
    AnionType,
    COMMON_ANION_TYPES,
    DEFAULT_PRECURSOR_ANIONS,
    METATHESIS_ANIONS,
    METATHESIS_COUNTER_CATIONS,
    generate_precursor_formula,
    generate_practical_precursors,
    generate_metathesis_sources,
    get_anion_by_name,
    get_oxidation_states,
    get_expanded_elements,
    get_elements_from_anion_types,
    get_practical_precursor_set,
    RecipeTemplate,
    generate_recipe_templates,
    suggest_recipes,
    filter_practical_templates,
    score_template_practicality,
)


class TestAnionType:
    """Tests for AnionType dataclass."""

    def test_anion_type_creation(self):
        anion = AnionType("test", "X", -1, frozenset({"X"}))
        assert anion.name == "test"
        assert anion.formula == "X"
        assert anion.charge == -1
        assert anion.elements == frozenset({"X"})

    def test_anion_type_frozen(self):
        anion = AnionType("test", "X", -1, frozenset({"X"}))
        with pytest.raises(AttributeError):
            anion.name = "changed"

    def test_anion_type_positive_charge_fails(self):
        with pytest.raises(ValueError, match="must be negative"):
            AnionType("bad", "X", 1, frozenset({"X"}))

    def test_common_anion_types_exist(self):
        assert len(COMMON_ANION_TYPES) > 0
        names = [a.name for a in COMMON_ANION_TYPES]
        assert "oxide" in names
        assert "carbonate" in names
        assert "chloride" in names

    def test_get_anion_by_name(self):
        oxide = get_anion_by_name("oxide")
        assert oxide.formula == "O"
        assert oxide.charge == -2

        carbonate = get_anion_by_name("carbonate")
        assert carbonate.formula == "CO3"
        assert carbonate.charge == -2

    def test_get_anion_by_name_unknown(self):
        with pytest.raises(ValueError, match="Unknown anion type"):
            get_anion_by_name("nonexistent")


class TestOxidationStates:
    """Tests for oxidation state lookup."""

    def test_get_oxidation_states_ba(self):
        states = get_oxidation_states("Ba")
        assert 2 in states
        assert all(s > 0 for s in states)

    def test_get_oxidation_states_fe(self):
        states = get_oxidation_states("Fe")
        assert 2 in states
        assert 3 in states

    def test_get_oxidation_states_ti(self):
        states = get_oxidation_states("Ti")
        assert 4 in states


class TestPrecursorFormulaGeneration:
    """Tests for generate_precursor_formula."""

    def test_simple_oxide_2plus(self):
        oxide = get_anion_by_name("oxide")
        formula = generate_precursor_formula("Ba", 2, oxide)
        assert formula == "BaO"

    def test_simple_oxide_3plus(self):
        oxide = get_anion_by_name("oxide")
        formula = generate_precursor_formula("Fe", 3, oxide)
        assert formula == "Fe2O3"

    def test_simple_oxide_4plus(self):
        oxide = get_anion_by_name("oxide")
        formula = generate_precursor_formula("Ti", 4, oxide)
        assert formula == "TiO2"

    def test_carbonate_2plus(self):
        carbonate = get_anion_by_name("carbonate")
        formula = generate_precursor_formula("Ba", 2, carbonate)
        assert formula == "BaCO3"

    def test_nitrate_2plus(self):
        nitrate = get_anion_by_name("nitrate")
        formula = generate_precursor_formula("Ba", 2, nitrate)
        assert formula == "Ba(NO3)2"

    def test_nitrate_1plus(self):
        nitrate = get_anion_by_name("nitrate")
        formula = generate_precursor_formula("Na", 1, nitrate)
        assert formula == "NaNO3"

    def test_hydroxide_2plus(self):
        hydroxide = get_anion_by_name("hydroxide")
        formula = generate_precursor_formula("Ba", 2, hydroxide)
        assert formula == "Ba(OH)2"

    def test_chloride_2plus(self):
        chloride = get_anion_by_name("chloride")
        formula = generate_precursor_formula("Ba", 2, chloride)
        assert formula == "BaCl2"

    def test_phosphate_3plus(self):
        phosphate = get_anion_by_name("phosphate")
        formula = generate_precursor_formula("Fe", 3, phosphate)
        assert formula == "FePO4"

    def test_invalid_oxidation_state(self):
        oxide = get_anion_by_name("oxide")
        with pytest.raises(ValueError, match="must be positive"):
            generate_precursor_formula("Ba", -2, oxide)
        with pytest.raises(ValueError, match="must be positive"):
            generate_precursor_formula("Ba", 0, oxide)


class TestGeneratePracticalPrecursors:
    """Tests for generate_practical_precursors."""

    def test_ba_precursors(self):
        precursors = generate_practical_precursors("Ba")
        assert "BaO" in precursors
        assert "BaCO3" in precursors
        assert "Ba(OH)2" in precursors
        assert "Ba(NO3)2" in precursors

    def test_fe_precursors_multiple_oxidation_states(self):
        precursors = generate_practical_precursors("Fe")
        # Fe2+ compounds
        assert "FeO" in precursors
        assert "FeCO3" in precursors
        # Fe3+ compounds
        assert "Fe2O3" in precursors

    def test_custom_anion_types(self):
        precursors = generate_practical_precursors("Ba", anion_types=["oxide", "chloride"])
        assert "BaO" in precursors
        assert "BaCl2" in precursors
        assert "BaCO3" not in precursors  # carbonate not in list

    def test_no_duplicates(self):
        precursors = generate_practical_precursors("Ba")
        assert len(precursors) == len(set(precursors))


class TestGenerateMetathesisSources:
    """Tests for generate_metathesis_sources."""

    def test_carbonate_sources(self):
        sources = generate_metathesis_sources("carbonate")
        assert "Na2CO3" in sources
        assert "K2CO3" in sources

    def test_hydroxide_sources(self):
        sources = generate_metathesis_sources("hydroxide")
        assert "NaOH" in sources
        assert "KOH" in sources

    def test_custom_counter_cations(self):
        sources = generate_metathesis_sources("chloride", ["Na", "Li"])
        assert "NaCl" in sources
        assert "LiCl" in sources
        assert "KCl" not in sources

    def test_ammonium_sources(self):
        sources = generate_metathesis_sources("chloride", ["NH4"])
        assert "NH4Cl" in sources


class TestGetExpandedElements:
    """Tests for get_expanded_elements."""

    def test_batio3_default(self):
        elements = get_expanded_elements("BaTiO3")
        # Should include target elements
        assert "Ba" in elements
        assert "Ti" in elements
        assert "O" in elements
        # Should include carbonate/nitrate/hydroxide elements
        assert "C" in elements
        assert "N" in elements
        assert "H" in elements
        # Should include metathesis counter-cations
        assert "Na" in elements
        assert "K" in elements

    def test_batio3_oxide_only(self):
        elements = get_expanded_elements("BaTiO3", anion_types=["oxide"], include_metathesis=False)
        assert elements == {"Ba", "Ti", "O"}

    def test_batio3_no_metathesis(self):
        elements = get_expanded_elements("BaTiO3", include_metathesis=False)
        # Should include target + anion elements
        assert "Ba" in elements
        assert "Ti" in elements
        assert "O" in elements
        assert "C" in elements  # from carbonate
        # Should NOT include metathesis stuff
        assert "Cl" not in elements
        assert "Na" not in elements

    def test_elements_from_anion_types(self):
        elements = get_elements_from_anion_types(["oxide", "carbonate"])
        assert "O" in elements
        assert "C" in elements
        assert "Cl" not in elements


class TestGetPracticalPrecursorSet:
    """Tests for get_practical_precursor_set."""

    def test_ba_ti_system(self):
        practical = get_practical_precursor_set({"Ba", "Ti", "O"})
        assert "BaO" in practical
        assert "BaCO3" in practical
        assert "TiO2" in practical

    def test_excludes_anion_elements(self):
        # O, C, N, H should not generate precursors themselves
        practical = get_practical_precursor_set({"Ba", "O", "C"})
        assert "BaO" in practical
        # Should not have precursors starting with O or C as cation


class TestRecipeTemplateGeneration:
    """Tests for recipe template generation."""

    def test_generate_recipe_templates_basic(self):
        available = ["BaO", "TiO2", "BaTiO3"]
        templates = generate_recipe_templates("BaTiO3", available, n_precursors=2)

        # Should find BaO + TiO2 as valid
        precursor_sets = [set(t.precursors) for t in templates]
        assert {"BaO", "TiO2"} in precursor_sets

    def test_generate_recipe_templates_excludes_target(self):
        available = ["BaO", "TiO2", "BaTiO3"]
        templates = generate_recipe_templates("BaTiO3", available, n_precursors=2)

        # Target should not be in any precursor set
        for t in templates:
            assert "BaTiO3" not in t.precursors

    def test_suggest_recipes_filters_practical(self):
        # Use Ba2TiO4 as a less practical intermediate phase
        available = ["BaO", "BaCO3", "TiO2", "Ti2O3", "Ba2TiO4", "BaTiO3"]
        templates = suggest_recipes("BaTiO3", available, practical_only=True)

        # Should prioritize practical combinations
        assert len(templates) > 0
        # BaO + TiO2 should be ranked highly
        top_precursors = set(templates[0].precursors)
        assert "TiO2" in top_precursors or "BaO" in top_precursors


class TestPracticalityScoring:
    """Tests for practicality scoring functions."""

    def test_score_template_practicality_all_practical(self):
        template = RecipeTemplate(
            precursors=["BaO", "TiO2"],
            target_phase="BaTiO3",
        )
        score = score_template_practicality(template)
        assert score == 1.0

    def test_score_template_practicality_mixed(self):
        # Ba2TiO4 is a real phase but not a "practical" precursor
        template = RecipeTemplate(
            precursors=["BaO", "Ba2TiO4"],
            target_phase="BaTiO3",
        )
        score = score_template_practicality(template)
        # BaO is practical, Ba2TiO4 is not (it's an intermediate, not a precursor)
        assert score == 0.5

    def test_filter_practical_templates(self):
        templates = [
            RecipeTemplate(precursors=["BaO", "TiO2"], target_phase="BaTiO3"),
            RecipeTemplate(precursors=["BaO", "Ba2TiO4"], target_phase="BaTiO3"),
            RecipeTemplate(precursors=["Ba2TiO4", "Ti2O3"], target_phase="BaTiO3"),
        ]
        filtered = filter_practical_templates(templates, min_practicality=0.5)

        # Should keep first two, reject third (both Ba2TiO4 and Ti2O3 are impractical)
        assert len(filtered) == 2
        # Should be sorted by practicality (highest first)
        assert filtered[0].metadata["practicality_score"] == 1.0
        assert filtered[1].metadata["practicality_score"] == 0.5
