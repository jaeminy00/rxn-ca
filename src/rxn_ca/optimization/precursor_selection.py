"""Precursor selection and recipe template generation.

This module provides utilities for automatically generating valid precursor
combinations (recipe templates) for a target phase based on element coverage
and stoichiometric constraints.

Hypotheses to explore:
1. Recipe templates can be computed by finding phase combinations that cover
   required elements (excluding O which comes from oxides)
2. Number of precursors (2 or 3) should be a hyperparameter
3. Stoichiometric feasibility can filter impractical combinations
4. Practicality scoring can prioritize common/stable precursors
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from pymatgen.core import Composition


@dataclass
class RecipeTemplate:
    """A candidate precursor combination for synthesizing a target phase.

    Attributes:
        precursors: List of precursor formulas
        target_phase: The target product phase
        covered_elements: Elements provided by this combination
        metadata: Additional info (e.g., stoichiometry notes, practicality score)
    """
    precursors: List[str]
    target_phase: str
    covered_elements: Set[str] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RecipeTemplate({' + '.join(self.precursors)} → {self.target_phase})"

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.precursors)), self.target_phase))

    def __eq__(self, other) -> bool:
        if not isinstance(other, RecipeTemplate):
            return False
        return (
            set(self.precursors) == set(other.precursors)
            and self.target_phase == other.target_phase
        )


def get_required_elements(target_phase: str, exclude_oxygen: bool = True) -> Set[str]:
    """Get the elements that must be provided by precursors.

    Args:
        target_phase: Chemical formula of target (e.g., "BaTiO3")
        exclude_oxygen: If True, exclude O (assumes it comes from oxide precursors)

    Returns:
        Set of element symbols that must be covered
    """
    comp = Composition(target_phase)
    elements = {str(el) for el in comp.elements}

    if exclude_oxygen:
        elements.discard("O")

    return elements


def get_phase_elements(phase: str) -> Set[str]:
    """Get elements present in a phase.

    Args:
        phase: Chemical formula

    Returns:
        Set of element symbols
    """
    comp = Composition(phase)
    return {str(el) for el in comp.elements}


def covers_required_elements(
    precursors: List[str],
    required_elements: Set[str],
) -> bool:
    """Check if a precursor combination covers all required elements.

    Args:
        precursors: List of precursor formulas
        required_elements: Elements that must be present

    Returns:
        True if the combination covers all required elements
    """
    covered = set()
    for precursor in precursors:
        covered.update(get_phase_elements(precursor))

    return required_elements.issubset(covered)


def generate_recipe_templates(
    target_phase: str,
    available_phases: List[str],
    n_precursors: int = 2,
    exclude_oxygen: bool = True,
    exclude_target: bool = True,
) -> List[RecipeTemplate]:
    """Generate valid precursor combinations for a target phase.

    A valid combination is one where the precursors collectively provide
    all required elements of the target phase.

    Args:
        target_phase: Chemical formula of target product (e.g., "BaTiO3")
        available_phases: List of available phase formulas
        n_precursors: Number of precursors in each template (2 or 3)
        exclude_oxygen: Exclude O from required elements (comes from oxides)
        exclude_target: Exclude the target phase itself from precursors

    Returns:
        List of valid RecipeTemplate objects
    """
    required = get_required_elements(target_phase, exclude_oxygen=exclude_oxygen)

    # Filter available phases
    candidates = list(available_phases)
    if exclude_target:
        candidates = [p for p in candidates if p != target_phase]

    templates = []
    for combo in combinations(candidates, n_precursors):
        if covers_required_elements(list(combo), required):
            covered = set()
            for p in combo:
                covered.update(get_phase_elements(p))

            template = RecipeTemplate(
                precursors=list(combo),
                target_phase=target_phase,
                covered_elements=covered,
            )
            templates.append(template)

    return templates


def filter_by_element_sources(
    templates: List[RecipeTemplate],
    required_elements: Set[str],
    max_sources_per_element: int = 2,
) -> List[RecipeTemplate]:
    """Filter templates where too many precursors provide the same element.

    For example, if we need Ba and Ti, having 3 Ba-containing precursors
    is probably not useful.

    Args:
        templates: List of RecipeTemplates to filter
        required_elements: Elements that matter for this check
        max_sources_per_element: Maximum precursors providing same element

    Returns:
        Filtered list of templates
    """
    filtered = []
    for template in templates:
        # Count how many precursors provide each required element
        element_counts = {el: 0 for el in required_elements}

        for precursor in template.precursors:
            precursor_elements = get_phase_elements(precursor)
            for el in required_elements:
                if el in precursor_elements:
                    element_counts[el] += 1

        # Check if any element is over-sourced
        if all(count <= max_sources_per_element for count in element_counts.values()):
            filtered.append(template)

    return filtered


# Common/practical precursors for various elements
# These are phases that are typically used in solid-state synthesis
PRACTICAL_PRECURSORS = {
    "Ba": ["BaO", "BaCO3", "BaO2", "Ba(OH)2"],
    "Ti": ["TiO2", "Ti2O3"],
    "Sr": ["SrO", "SrCO3", "Sr(OH)2"],
    "Ca": ["CaO", "CaCO3", "Ca(OH)2"],
    "Pb": ["PbO", "PbO2"],
    "Zr": ["ZrO2"],
    "Fe": ["Fe2O3", "Fe3O4", "FeO"],
    "Co": ["Co3O4", "CoO"],
    "Ni": ["NiO"],
    "Mn": ["MnO2", "Mn2O3", "MnO"],
    "Al": ["Al2O3"],
    "Si": ["SiO2"],
    "Mg": ["MgO"],
    "Zn": ["ZnO"],
    "Cu": ["CuO", "Cu2O"],
    "Li": ["Li2O", "Li2CO3"],
    "Na": ["Na2O", "Na2CO3", "NaCl"],
    "K": ["K2O", "K2CO3", "KCl"],
}


def score_template_practicality(
    template: RecipeTemplate,
    practical_precursors: Optional[Dict[str, List[str]]] = None,
) -> float:
    """Score a template based on how practical/common its precursors are.

    Higher score = more practical precursors.

    Args:
        template: RecipeTemplate to score
        practical_precursors: Dict mapping elements to practical precursor lists.
            Defaults to PRACTICAL_PRECURSORS.

    Returns:
        Practicality score (0.0 to 1.0)
    """
    if practical_precursors is None:
        practical_precursors = PRACTICAL_PRECURSORS

    # Flatten practical precursors into a set
    practical_set = set()
    for precursors in practical_precursors.values():
        practical_set.update(precursors)

    # Count how many precursors are in the practical set
    practical_count = sum(1 for p in template.precursors if p in practical_set)

    return practical_count / len(template.precursors)


def filter_practical_templates(
    templates: List[RecipeTemplate],
    min_practicality: float = 0.5,
    practical_precursors: Optional[Dict[str, List[str]]] = None,
) -> List[RecipeTemplate]:
    """Filter templates to keep only those with practical precursors.

    Args:
        templates: List of RecipeTemplates to filter
        min_practicality: Minimum practicality score (0.0 to 1.0)
        practical_precursors: Dict mapping elements to practical precursor lists

    Returns:
        Filtered list of templates with practicality scores in metadata
    """
    filtered = []
    for template in templates:
        score = score_template_practicality(template, practical_precursors)
        if score >= min_practicality:
            template.metadata["practicality_score"] = score
            filtered.append(template)

    # Sort by practicality score (highest first)
    filtered.sort(key=lambda t: t.metadata.get("practicality_score", 0), reverse=True)

    return filtered


def get_stoichiometry_ratio(
    precursors: List[str],
    target_phase: str,
    element: str,
) -> Optional[float]:
    """Calculate what ratio of precursors would achieve target stoichiometry for one element.

    This is a simplified calculation that doesn't account for reaction balancing,
    but gives a rough sense of feasibility.

    Args:
        precursors: List of precursor formulas
        target_phase: Target phase formula
        element: Element to check ratio for

    Returns:
        Ratio needed, or None if element not in precursors
    """
    target_comp = Composition(target_phase)
    target_amt = target_comp.get(element, 0)

    if target_amt == 0:
        return None

    # Find which precursors contain this element
    precursor_amts = []
    for p in precursors:
        p_comp = Composition(p)
        amt = p_comp.get(element, 0)
        if amt > 0:
            precursor_amts.append((p, amt))

    if not precursor_amts:
        return None

    # If only one precursor has this element, ratio is determined
    if len(precursor_amts) == 1:
        return target_amt / precursor_amts[0][1]

    # Multiple precursors - more complex, return None for now
    return None


def analyze_template_stoichiometry(
    template: RecipeTemplate,
) -> Dict[str, any]:
    """Analyze stoichiometric relationships in a template.

    Args:
        template: RecipeTemplate to analyze

    Returns:
        Dict with stoichiometry analysis
    """
    target_comp = Composition(template.target_phase)
    required = get_required_elements(template.target_phase)

    analysis = {
        "target_composition": dict(target_comp.as_dict()),
        "required_elements": list(required),
        "element_sources": {},
    }

    for el in required:
        sources = []
        for p in template.precursors:
            p_comp = Composition(p)
            if p_comp.get(el, 0) > 0:
                sources.append({
                    "precursor": p,
                    "amount_per_formula": p_comp.get(el),
                })
        analysis["element_sources"][el] = sources

    template.metadata["stoichiometry"] = analysis
    return analysis


# ============================================================================
# Literature-grounded scoring (using text-mined synthesis data)
# ============================================================================

def score_template_by_literature(
    template: RecipeTemplate,
    synthesis_dataset: "SynthesisDataset",
    method: str = "geometric",
) -> float:
    """Score a template based on literature frequency of its precursors.

    This provides a data-driven alternative to the hardcoded PRACTICAL_PRECURSORS.

    Args:
        template: RecipeTemplate to score
        synthesis_dataset: SynthesisDataset instance with literature data
        method: Scoring method ("frequency", "pair", or "geometric")

    Returns:
        Score based on literature frequency (higher = more commonly used)
    """
    return synthesis_dataset.score_precursor_set(template.precursors, method=method)


def filter_templates_by_literature(
    templates: List[RecipeTemplate],
    synthesis_dataset: "SynthesisDataset",
    min_frequency: int = 1,
    scoring_method: str = "geometric",
) -> List[RecipeTemplate]:
    """Filter and rank templates based on literature frequency.

    Args:
        templates: List of RecipeTemplates to filter
        synthesis_dataset: SynthesisDataset instance
        min_frequency: Minimum times each precursor must appear in literature
        scoring_method: How to score templates ("frequency", "pair", "geometric")

    Returns:
        Filtered and sorted templates with literature_score in metadata
    """
    filtered = []
    for template in templates:
        # Check if all precursors meet minimum frequency
        all_frequent = all(
            synthesis_dataset.get_precursor_frequency(p) >= min_frequency
            for p in template.precursors
        )

        if all_frequent or min_frequency == 0:
            score = score_template_by_literature(
                template, synthesis_dataset, method=scoring_method
            )
            template.metadata["literature_score"] = score
            template.metadata["precursor_frequencies"] = {
                p: synthesis_dataset.get_precursor_frequency(p)
                for p in template.precursors
            }
            filtered.append(template)

    # Sort by literature score (highest first)
    filtered.sort(key=lambda t: t.metadata.get("literature_score", 0), reverse=True)

    return filtered


def get_practical_precursors_from_literature(
    synthesis_dataset: "SynthesisDataset",
    elements: List[str],
    min_frequency: int = 5,
    max_per_element: int = 5,
) -> Dict[str, List[str]]:
    """Get practical precursors for elements based on literature frequency.

    This replaces the hardcoded PRACTICAL_PRECURSORS with data-driven selection.

    Args:
        synthesis_dataset: SynthesisDataset instance
        elements: Elements to find precursors for (e.g., ["Ba", "Ti"])
        min_frequency: Minimum literature occurrences
        max_per_element: Max precursors to return per element

    Returns:
        Dict mapping element -> list of practical precursor formulas
    """
    result = {}
    for el in elements:
        top_precursors = synthesis_dataset.get_precursors_for_element(el, n=max_per_element * 2)
        filtered = [(p, count) for p, count in top_precursors if count >= min_frequency]
        result[el] = [p for p, _ in filtered[:max_per_element]]
    return result


# ============================================================================
# Main entry points
# ============================================================================

# Convenience function for common use case
def suggest_recipes(
    target_phase: str,
    available_phases: List[str],
    n_precursors: int = 2,
    practical_only: bool = True,
    max_templates: int = 10,
) -> List[RecipeTemplate]:
    """Suggest practical recipe templates for a target phase.

    This is the main entry point for recipe template generation.

    Args:
        target_phase: Chemical formula of target (e.g., "BaTiO3")
        available_phases: List of available phase formulas
        n_precursors: Number of precursors (2 or 3)
        practical_only: Filter to practical precursors only
        max_templates: Maximum number of templates to return

    Returns:
        List of RecipeTemplate objects, sorted by practicality
    """
    # Generate all valid templates
    templates = generate_recipe_templates(
        target_phase=target_phase,
        available_phases=available_phases,
        n_precursors=n_precursors,
    )

    # Filter by element distribution
    required = get_required_elements(target_phase)
    templates = filter_by_element_sources(templates, required)

    # Filter by practicality if requested
    if practical_only:
        templates = filter_practical_templates(templates, min_practicality=0.5)
    else:
        # Still score them for sorting
        for t in templates:
            t.metadata["practicality_score"] = score_template_practicality(t)
        templates.sort(key=lambda t: t.metadata.get("practicality_score", 0), reverse=True)

    # Analyze stoichiometry for top templates
    for t in templates[:max_templates]:
        analyze_template_stoichiometry(t)

    return templates[:max_templates]


def suggest_recipes_from_literature(
    target_phase: str,
    available_phases: List[str],
    synthesis_dataset: "SynthesisDataset",
    n_precursors: int = 2,
    min_frequency: int = 5,
    max_templates: int = 10,
) -> List[RecipeTemplate]:
    """Suggest recipe templates grounded in literature synthesis data.

    This is the data-driven alternative to suggest_recipes() that uses
    text-mined synthesis frequencies instead of hardcoded precursor lists.

    Args:
        target_phase: Chemical formula of target (e.g., "BaTiO3")
        available_phases: List of available phase formulas in the system
        synthesis_dataset: SynthesisDataset instance with literature data
        n_precursors: Number of precursors (2 or 3)
        min_frequency: Minimum times each precursor must appear in literature
        max_templates: Maximum number of templates to return

    Returns:
        List of RecipeTemplate objects, sorted by literature frequency
    """
    # Generate all valid templates
    templates = generate_recipe_templates(
        target_phase=target_phase,
        available_phases=available_phases,
        n_precursors=n_precursors,
    )

    # Filter by element distribution
    required = get_required_elements(target_phase)
    templates = filter_by_element_sources(templates, required)

    # Filter and rank by literature frequency
    templates = filter_templates_by_literature(
        templates,
        synthesis_dataset,
        min_frequency=min_frequency,
        scoring_method="geometric",
    )

    # Analyze stoichiometry for top templates
    for t in templates[:max_templates]:
        analyze_template_stoichiometry(t)

    return templates[:max_templates]
