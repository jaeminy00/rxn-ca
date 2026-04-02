"""Precursor selection and recipe template generation.

This module provides utilities for automatically generating valid precursor
combinations (recipe templates) for a target phase based on element coverage
and stoichiometric constraints.

Key features:
1. Programmatic precursor generation from oxidation states and anion types
2. Element coverage analysis for recipe templates
3. Support for metathesis (salt exchange) reactions
4. Integration with literature data and thermodynamic scoring
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import gcd
from typing import Dict, List, Optional, Set, Tuple

from pymatgen.core import Composition, Element


# =============================================================================
# Anion types for precursor generation
# =============================================================================


@dataclass(frozen=True)
class AnionType:
    """Definition of an anion type for precursor generation.

    Attributes:
        name: Human-readable name (e.g., "carbonate")
        formula: Chemical formula of the anion (e.g., "CO3")
        charge: Ionic charge (negative integer, e.g., -2)
        elements: Set of elements introduced by this anion (for element expansion)
    """

    name: str
    formula: str
    charge: int
    elements: frozenset

    def __post_init__(self):
        if self.charge >= 0:
            raise ValueError(f"Anion charge must be negative, got {self.charge}")


# Common anion types for solid-state synthesis precursors
COMMON_ANION_TYPES: List[AnionType] = [
    # Direct precursors (decompose to give oxide + gas)
    AnionType("oxide", "O", -2, frozenset({"O"})),
    AnionType("peroxide", "O2", -2, frozenset({"O"})),
    AnionType("carbonate", "CO3", -2, frozenset({"C", "O"})),
    AnionType("nitrate", "NO3", -1, frozenset({"N", "O"})),
    AnionType("hydroxide", "OH", -1, frozenset({"O", "H"})),
    AnionType("acetate", "C2H3O2", -1, frozenset({"C", "H", "O"})),
    # Metathesis salts (halides)
    AnionType("chloride", "Cl", -1, frozenset({"Cl"})),
    AnionType("bromide", "Br", -1, frozenset({"Br"})),
    AnionType("iodide", "I", -1, frozenset({"I"})),
    AnionType("fluoride", "F", -1, frozenset({"F"})),
    # Other useful anions
    AnionType("sulfate", "SO4", -2, frozenset({"S", "O"})),
    AnionType("oxalate", "C2O4", -2, frozenset({"C", "O"})),
    AnionType("phosphate", "PO4", -3, frozenset({"P", "O"})),
]

# Default anion types for typical solid-state synthesis (by formula)
DEFAULT_PRECURSOR_ANIONS: List[str] = ["O", "CO3", "OH", "NO3"]

# Anion types useful for metathesis reactions (by formula)
METATHESIS_ANIONS: List[str] = ["Cl", "Br", "NO3", "SO4", "C2H3O2"]

# Counter-cations for metathesis reactions (provide leaving groups)
# Maps cation symbol to its oxidation state
METATHESIS_COUNTER_CATIONS: Dict[str, int] = {
    "Na": 1,
    "K": 1,
    "Li": 1,
    "NH4": 1,  # Ammonium - decomposes to NH3 + H+
    "Cs": 1,
}

# Build formula -> AnionType lookup (populated after COMMON_ANION_TYPES is defined)
_ANION_BY_FORMULA: Dict[str, "AnionType"] = {}


# =============================================================================
# Precursor formula generation
# =============================================================================


def get_oxidation_states(element: str) -> Tuple[int, ...]:
    """Get common oxidation states for an element using pymatgen.

    Args:
        element: Element symbol (e.g., "Fe", "Ti")

    Returns:
        Tuple of common positive oxidation states

    Examples:
        >>> get_oxidation_states("Fe")
        (2, 3)
        >>> get_oxidation_states("Ba")
        (2,)
    """
    el = Element(element)
    # Filter to positive oxidation states only (for cations)
    return tuple(ox for ox in el.common_oxidation_states if ox > 0)


def generate_precursor_formula(
    cation: str,
    oxidation_state: int,
    anion: AnionType,
) -> str:
    """Generate a charge-balanced precursor formula.

    Args:
        cation: Cation symbol (e.g., "Ba", "Fe", "NH4")
        oxidation_state: Positive oxidation state of cation
        anion: AnionType instance

    Returns:
        Charge-balanced formula string (e.g., "BaCO3", "Fe2O3", "Ba(NO3)2")

    Examples:
        >>> oxide = AnionType("oxide", "O", -2, frozenset({"O"}))
        >>> generate_precursor_formula("Ba", 2, oxide)
        'BaO'
        >>> generate_precursor_formula("Fe", 3, oxide)
        'Fe2O3'
        >>> nitrate = AnionType("nitrate", "NO3", -1, frozenset({"N", "O"}))
        >>> generate_precursor_formula("Ba", 2, nitrate)
        'Ba(NO3)2'
    """
    if oxidation_state <= 0:
        raise ValueError(f"Oxidation state must be positive, got {oxidation_state}")

    # Calculate stoichiometric coefficients to balance charges
    # n_cation * ox_state + n_anion * anion_charge = 0
    anion_charge_abs = abs(anion.charge)
    g = gcd(oxidation_state, anion_charge_abs)
    n_cation = anion_charge_abs // g
    n_anion = oxidation_state // g

    # Format cation part
    if n_cation == 1:
        cation_str = cation
    else:
        cation_str = f"{cation}{n_cation}"

    # Format anion part
    # Polyatomic ions (more than 2 chars or contains uppercase after first) need parentheses
    is_polyatomic = len(anion.formula) > 2 or any(
        c.isupper() for c in anion.formula[1:]
    )

    if n_anion == 1:
        anion_str = anion.formula
    elif is_polyatomic:
        anion_str = f"({anion.formula}){n_anion}"
    else:
        anion_str = f"{anion.formula}{n_anion}"

    return cation_str + anion_str


def _build_anion_lookup() -> None:
    """Build the formula -> AnionType lookup dict."""
    for anion in COMMON_ANION_TYPES:
        _ANION_BY_FORMULA[anion.formula] = anion


def get_anion(identifier: str) -> AnionType:
    """Get an AnionType by its formula or name.

    Args:
        identifier: Anion formula (e.g., "CO3", "Cl") or name (e.g., "carbonate")

    Returns:
        Matching AnionType

    Raises:
        ValueError: If anion not found

    Examples:
        >>> get_anion("CO3")
        AnionType(name='carbonate', formula='CO3', ...)
        >>> get_anion("Cl")
        AnionType(name='chloride', formula='Cl', ...)
        >>> get_anion("carbonate")  # name also works for backwards compat
        AnionType(name='carbonate', formula='CO3', ...)
    """
    # Ensure lookup is populated
    if not _ANION_BY_FORMULA:
        _build_anion_lookup()

    # Try formula first
    if identifier in _ANION_BY_FORMULA:
        return _ANION_BY_FORMULA[identifier]

    # Fall back to name lookup for backwards compatibility
    for anion in COMMON_ANION_TYPES:
        if anion.name == identifier:
            return anion

    raise ValueError(
        f"Unknown anion: '{identifier}'. "
        f"Valid formulas: {list(_ANION_BY_FORMULA.keys())}"
    )


def generate_practical_precursors(
    element: str,
    anion_types: Optional[List[str]] = None,
    oxidation_states: Optional[List[int]] = None,
) -> List[str]:
    """Generate practical precursor formulas for a cation element.

    Uses pymatgen's common_oxidation_states unless overridden.

    Args:
        element: Cation element symbol (e.g., "Ba", "Ti", "Fe")
        anion_types: List of anion type names to use. Defaults to DEFAULT_PRECURSOR_ANIONS.
        oxidation_states: Specific oxidation states to use. Defaults to pymatgen's common states.

    Returns:
        List of precursor formula strings

    Examples:
        >>> generate_practical_precursors("Ba")
        ['BaO', 'BaCO3', 'Ba(OH)2', 'Ba(NO3)2']
        >>> generate_practical_precursors("Fe")
        ['FeO', 'Fe2O3', 'FeCO3', 'Fe2(CO3)3', 'Fe(OH)2', 'Fe(OH)3', ...]
    """
    if anion_types is None:
        anion_types = DEFAULT_PRECURSOR_ANIONS

    if oxidation_states is None:
        oxidation_states = list(get_oxidation_states(element))

    if not oxidation_states:
        return []

    anions = [get_anion(a) for a in anion_types]

    precursors = []
    seen = set()
    for ox_state in oxidation_states:
        for anion in anions:
            formula = generate_precursor_formula(element, ox_state, anion)
            if formula not in seen:
                precursors.append(formula)
                seen.add(formula)

    return precursors


def generate_metathesis_sources(
    target_anion: str,
    counter_cations: Optional[List[str]] = None,
) -> List[str]:
    """Generate counter-cation sources for metathesis reactions.

    These are alkali/ammonium salts that provide anions via double displacement.
    E.g., Na2CO3 reacts with BaCl2 to precipitate BaCO3.

    Args:
        target_anion: Name of the anion to source (e.g., "carbonate", "oxide")
        counter_cations: Cations to use. Defaults to ["Na", "K"].

    Returns:
        List of counter-cation salt formulas

    Examples:
        >>> generate_metathesis_sources("carbonate")
        ['Na2CO3', 'K2CO3']
        >>> generate_metathesis_sources("hydroxide", ["Na", "K", "Li"])
        ['NaOH', 'KOH', 'LiOH']
    """
    if counter_cations is None:
        counter_cations = ["Na", "K"]

    anion = get_anion(target_anion)

    sources = []
    for cation in counter_cations:
        if cation not in METATHESIS_COUNTER_CATIONS:
            raise ValueError(f"Unknown counter-cation: {cation}")
        ox_state = METATHESIS_COUNTER_CATIONS[cation]
        formula = generate_precursor_formula(cation, ox_state, anion)
        sources.append(formula)

    return sources


def get_elements_from_anions(anions: Optional[List[str]] = None) -> Set[str]:
    """Get all elements introduced by a set of anions.

    Args:
        anions: List of anion formulas (e.g., ["CO3", "Cl"]).
            Defaults to DEFAULT_PRECURSOR_ANIONS.

    Returns:
        Set of element symbols
    """
    if anions is None:
        anions = DEFAULT_PRECURSOR_ANIONS

    elements: Set[str] = set()
    for identifier in anions:
        anion = get_anion(identifier)
        elements.update(anion.elements)

    return elements


def get_expanded_elements(
    target_phase: str,
    anions: Optional[List[str]] = None,
    metathesis_anions: Optional[List[str]] = None,
    counter_cations: Optional[List[str]] = None,
) -> Set[str]:
    """Get the full set of elements needed for precursor selection.

    This expands beyond the target phase elements to include elements from
    precursor anions (e.g., C from CO3, N from NO3) and optionally metathesis
    reagents.

    Use this to determine what elements to pass to get_entries().

    Args:
        target_phase: Target product formula (e.g., "BaTiO3")
        anions: Base anion formulas for precursors. Defaults to ["O", "CO3", "OH", "NO3"].
        metathesis_anions: Additional anions for metathesis (e.g., ["Cl"]).
            Defaults to None (no metathesis anions).
        counter_cations: Counter-cations for metathesis (e.g., ["Na", "K"]).
            Defaults to None (no counter-cations).

    Returns:
        Set of element symbols needed for get_entries()

    Examples:
        >>> get_expanded_elements("BaTiO3")
        {'Ba', 'Ti', 'O', 'C', 'N', 'H'}
        >>> get_expanded_elements("BaTiO3", anions=["O"])
        {'Ba', 'Ti', 'O'}
        >>> get_expanded_elements("BaTiO3", metathesis_anions=["Cl"], counter_cations=["Na"])
        {'Ba', 'Ti', 'O', 'C', 'N', 'H', 'Cl', 'Na'}
    """
    # Start with target phase elements
    target_comp = Composition(target_phase)
    elements = {str(el) for el in target_comp.elements}

    # Add elements from base precursor anions
    if anions is None:
        anions = list(DEFAULT_PRECURSOR_ANIONS)

    all_anions = list(anions)

    # Add metathesis anions if specified
    if metathesis_anions:
        all_anions = list(set(all_anions + metathesis_anions))

    elements.update(get_elements_from_anions(all_anions))

    # Add counter-cation elements if specified
    if counter_cations:
        for cation in counter_cations:
            if cation == "NH4":
                # NH4 is not a real element, add N and H instead
                elements.add("N")
                elements.add("H")
            elif cation not in METATHESIS_COUNTER_CATIONS:
                raise ValueError(
                    f"Unknown counter-cation: '{cation}'. "
                    f"Valid options: {list(METATHESIS_COUNTER_CATIONS.keys())}"
                )
            else:
                elements.add(cation)

    return elements


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


def get_practical_precursor_set(
    elements: Optional[Set[str]] = None,
    anion_types: Optional[List[str]] = None,
) -> Set[str]:
    """Generate the set of all practical precursors for given elements.

    Args:
        elements: Elements to generate precursors for. If None, uses common cations.
        anion_types: Anion types to use. Defaults to DEFAULT_PRECURSOR_ANIONS.

    Returns:
        Set of practical precursor formula strings
    """
    if elements is None:
        # Use a default set of common cations
        elements = {
            "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba",
            "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Al", "Si", "Zr", "Nb", "Mo", "Sn", "Pb", "Bi",
            "La", "Ce", "Nd", "Y",
        }

    practical_set: Set[str] = set()
    for el in elements:
        # Skip non-cation elements
        if el in {"O", "C", "N", "H", "S", "Cl", "Br", "I", "F", "P"}:
            continue
        try:
            precursors = generate_practical_precursors(el, anion_types=anion_types)
            practical_set.update(precursors)
        except (ValueError, KeyError):
            # Element doesn't have known oxidation states
            continue

    return practical_set


def score_template_practicality(
    template: RecipeTemplate,
    practical_set: Optional[Set[str]] = None,
    anion_types: Optional[List[str]] = None,
) -> float:
    """Score a template based on how practical/common its precursors are.

    A precursor is considered practical if it matches the formula pattern
    generated by generate_practical_precursors() for its cation element.

    Higher score = more practical precursors.

    Args:
        template: RecipeTemplate to score
        practical_set: Pre-computed set of practical precursors. If None,
            generates based on elements in the template.
        anion_types: Anion types to consider practical. Defaults to DEFAULT_PRECURSOR_ANIONS.

    Returns:
        Practicality score (0.0 to 1.0)
    """
    if practical_set is None:
        # Generate practical set based on elements in template
        template_elements: Set[str] = set()
        for precursor in template.precursors:
            template_elements.update(get_phase_elements(precursor))
        practical_set = get_practical_precursor_set(template_elements, anion_types)

    # Count how many precursors are in the practical set
    practical_count = sum(1 for p in template.precursors if p in practical_set)

    return practical_count / len(template.precursors)


def filter_practical_templates(
    templates: List[RecipeTemplate],
    min_practicality: float = 0.5,
    anion_types: Optional[List[str]] = None,
) -> List[RecipeTemplate]:
    """Filter templates to keep only those with practical precursors.

    Args:
        templates: List of RecipeTemplates to filter
        min_practicality: Minimum practicality score (0.0 to 1.0)
        anion_types: Anion types to consider practical. Defaults to DEFAULT_PRECURSOR_ANIONS.

    Returns:
        Filtered list of templates with practicality scores in metadata
    """
    # Pre-compute practical set for all elements across templates
    all_elements: Set[str] = set()
    for template in templates:
        for precursor in template.precursors:
            all_elements.update(get_phase_elements(precursor))
    practical_set = get_practical_precursor_set(all_elements, anion_types)

    filtered = []
    for template in templates:
        score = score_template_practicality(template, practical_set)
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

    This provides data-driven precursor selection based on what has been
    observed in text-mined synthesis literature.

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
