"""Thermodynamic scoring for precursor selection using ARROWS.

This module integrates ARROWS (Autonomous Reaction Route Optimization with
Solid-State Synthesis) to provide thermodynamic ΔG-based scoring for recipe
templates, complementing the literature-based frequency scoring.

Key insight from Szymanski et al. (Nature Communications, 2023):
- Precursor sets with larger (more negative) ΔG have stronger driving force
- But intermediates can trap energy, reducing effective ΔG'
- ΔG' = ΔG_initial - ΔG_consumed_by_intermediates

Reference: https://doi.org/10.1038/s41467-023-42329-9
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.core import Composition


@dataclass
class ThermodynamicScore:
    """Result of thermodynamic scoring for a precursor set.

    Attributes:
        precursors: List of precursor formulas
        target: Target phase formula
        delta_g: Reaction energy in meV/atom (more negative = more favorable)
        temperature: Temperature at which ΔG was calculated (K)
        products: Expected products (may include byproducts)
        is_favorable: Whether ΔG < 0 (thermodynamically spontaneous)
    """
    precursors: List[str]
    target: str
    delta_g: float
    temperature: float
    products: List[str]
    is_favorable: bool

    def __repr__(self) -> str:
        sign = "" if self.delta_g < 0 else "+"
        return f"ThermodynamicScore({' + '.join(self.precursors)} → {self.target}, ΔG={sign}{self.delta_g:.1f} meV/atom)"


class ARROWSIntegration:
    """Integration with ARROWS for thermodynamic scoring.

    This class wraps ARROWS functionality to:
    1. Calculate ΔG for precursor → target reactions
    2. Build temperature-dependent phase diagrams
    3. Score recipe templates by thermodynamic driving force

    Example:
        >>> from rxn_ca.optimization.thermodynamic_scoring import ARROWSIntegration
        >>> arrows = ARROWSIntegration()
        >>> arrows.initialize(["BaO", "TiO2", "BaTiO3"], temps=[800, 1000, 1200])
        >>> score = arrows.get_reaction_energy(["BaO", "TiO2"], "BaTiO3", temp=1000)
        >>> print(f"ΔG = {score.delta_g} meV/atom")
    """

    def __init__(self):
        """Initialize ARROWS integration."""
        self._pd_dict: Optional[Dict[float, Any]] = None
        self._temperatures: List[float] = []
        self._available_phases: List[str] = []
        self._arrows_available = self._check_arrows_available()

    def _check_arrows_available(self) -> bool:
        """Check if ARROWS is importable."""
        try:
            from arrows import energetics, reactions
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Whether ARROWS is available for use."""
        return self._arrows_available

    def initialize(
        self,
        available_phases: List[str],
        temperatures: List[float],
        atmosphere: str = "air",
    ) -> None:
        """Initialize phase diagrams for the given chemical space.

        This is an expensive operation that queries the Materials Project
        and builds phase diagrams at each temperature. Call once per system.

        Args:
            available_phases: List of phase formulas in the system
            temperatures: List of temperatures (K) to build phase diagrams for
            atmosphere: "air" or "inert" (affects O2 partial pressure)
        """
        if not self._arrows_available:
            raise ImportError(
                "ARROWS not available. Install from: https://github.com/njszym/ARROWS"
            )

        from arrows import energetics

        self._available_phases = available_phases
        self._temperatures = sorted(temperatures)

        # Round temperatures to nearest 100 (ARROWS requirement)
        rounded_temps = [round(t, -2) for t in self._temperatures]
        unique_temps = sorted(set(rounded_temps))

        # Build phase diagrams (expensive - queries MP)
        self._pd_dict = energetics.get_pd_dict(
            available_phases,
            unique_temps,
            atmos=atmosphere
        )

    def get_reaction_energy(
        self,
        precursors: List[str],
        target: str,
        temperature: float,
        allowed_byproducts: Optional[List[str]] = None,
    ) -> ThermodynamicScore:
        """Calculate ΔG for a precursor → target reaction.

        Args:
            precursors: List of precursor formulas
            target: Target product formula
            temperature: Temperature in K
            allowed_byproducts: Optional phases that may form alongside target

        Returns:
            ThermodynamicScore with ΔG and reaction details
        """
        if not self._arrows_available:
            raise ImportError("ARROWS not available")

        if self._pd_dict is None:
            raise RuntimeError("Must call initialize() before get_reaction_energy()")

        from arrows import reactions

        # Round temperature to nearest 100
        temp_rounded = round(temperature, -2)
        if temp_rounded not in self._pd_dict:
            # Use closest available temperature
            temp_rounded = min(self._pd_dict.keys(), key=lambda t: abs(t - temperature))

        phase_diagram = self._pd_dict[temp_rounded]

        # Try to calculate reaction energy
        try:
            delta_g = reactions.get_rxn_energy(
                precursors,
                [target],
                temp_rounded,
                phase_diagram
            )
            products = [target]
        except Exception as e:
            # If reaction can't be balanced with just target, try with byproducts
            if allowed_byproducts:
                for byproduct in allowed_byproducts:
                    try:
                        delta_g = reactions.get_rxn_energy(
                            precursors,
                            [target, byproduct],
                            temp_rounded,
                            phase_diagram
                        )
                        products = [target, byproduct]
                        break
                    except Exception:
                        continue
                else:
                    # No valid reaction found
                    return ThermodynamicScore(
                        precursors=precursors,
                        target=target,
                        delta_g=float('inf'),
                        temperature=temperature,
                        products=[],
                        is_favorable=False,
                    )
            else:
                return ThermodynamicScore(
                    precursors=precursors,
                    target=target,
                    delta_g=float('inf'),
                    temperature=temperature,
                    products=[],
                    is_favorable=False,
                )

        return ThermodynamicScore(
            precursors=precursors,
            target=target,
            delta_g=delta_g,
            temperature=temperature,
            products=products,
            is_favorable=delta_g < 0,
        )

    def get_formation_energy(
        self,
        formula: str,
        temperature: float,
    ) -> Optional[float]:
        """Get formation energy of a phase at given temperature.

        Args:
            formula: Chemical formula
            temperature: Temperature in K

        Returns:
            Formation energy in eV/atom, or None if not available
        """
        if not self._arrows_available:
            raise ImportError("ARROWS not available")

        from arrows import energetics

        temp_rounded = round(temperature, -2)
        return energetics.get_entry_Ef(formula, temp_rounded)

    def score_recipe_template(
        self,
        template: "RecipeTemplate",
        temperature: float,
        allowed_byproducts: Optional[List[str]] = None,
    ) -> float:
        """Score a recipe template by thermodynamic driving force.

        Args:
            template: RecipeTemplate to score
            temperature: Temperature in K
            allowed_byproducts: Optional byproducts to consider

        Returns:
            Score based on ΔG (more negative = higher score)
            Returns -inf for unfavorable or unbalanceable reactions
        """
        result = self.get_reaction_energy(
            template.precursors,
            template.target_phase,
            temperature,
            allowed_byproducts,
        )

        # Store in template metadata
        template.metadata["delta_g"] = result.delta_g
        template.metadata["thermodynamic_score"] = -result.delta_g  # Flip sign for ranking
        template.metadata["is_favorable"] = result.is_favorable

        if not result.is_favorable:
            return float('-inf')

        # Return negative ΔG as score (more negative ΔG = higher score)
        return -result.delta_g


def get_precursor_sets_arrows(
    available_phases: List[str],
    target: str,
    allowed_byproducts: Optional[List[str]] = None,
    max_precursors: Optional[int] = None,
    allow_oxidation: bool = True,
) -> List[Tuple[List[str], List[str]]]:
    """Get stoichiometrically balanced precursor sets using ARROWS.

    This wraps arrows.searcher.get_precursor_sets() for convenience.

    Args:
        available_phases: Phases that can be used as precursors
        target: Target product formula
        allowed_byproducts: Phases allowed as secondary products
        max_precursors: Maximum precursors per set (default: Gibbs phase rule)
        allow_oxidation: Whether to allow O2/CO2 as reactants

    Returns:
        List of (precursors, products) tuples
    """
    try:
        from arrows.searcher import get_precursor_sets
    except ImportError:
        raise ImportError(
            "ARROWS not available. Install from: https://github.com/njszym/ARROWS"
        )

    return get_precursor_sets(
        available_phases,
        [target],
        allowed_byproducts=allowed_byproducts or [],
        max_pc=max_precursors,
        allow_oxidation=allow_oxidation,
    )


# =============================================================================
# Combined scoring: Literature + Thermodynamics
# =============================================================================

def score_template_combined(
    template: "RecipeTemplate",
    synthesis_dataset: "SynthesisDataset",
    arrows_integration: ARROWSIntegration,
    temperature: float,
    lit_weight: float = 0.5,
    thermo_weight: float = 0.5,
    allowed_byproducts: Optional[List[str]] = None,
) -> float:
    """Score a template combining literature frequency and thermodynamic ΔG.

    This implements the key insight that both historical precedent (what has
    worked before) and physics (what should work) are valuable signals.

    Args:
        template: RecipeTemplate to score
        synthesis_dataset: SynthesisDataset for literature scoring
        arrows_integration: Initialized ARROWSIntegration
        temperature: Temperature in K
        lit_weight: Weight for literature score (0-1)
        thermo_weight: Weight for thermodynamic score (0-1)
        allowed_byproducts: Byproducts to consider for ΔG calculation

    Returns:
        Combined score (higher = better)
    """
    from .precursor_selection import score_template_by_literature

    # Literature score (log-scale, typically 1-10)
    lit_score = score_template_by_literature(
        template, synthesis_dataset, method="geometric"
    )

    # Thermodynamic score
    thermo_result = arrows_integration.get_reaction_energy(
        template.precursors,
        template.target_phase,
        temperature,
        allowed_byproducts,
    )

    # Store both scores in metadata
    template.metadata["literature_score"] = lit_score
    template.metadata["delta_g"] = thermo_result.delta_g
    template.metadata["is_favorable"] = thermo_result.is_favorable

    # If thermodynamically unfavorable, heavily penalize
    if not thermo_result.is_favorable:
        return lit_score * lit_weight * 0.1  # 90% penalty

    # Normalize thermodynamic score to similar range as literature
    # Typical ΔG values: -50 to -500 meV/atom for favorable reactions
    # Map to 0-10 range: -500 meV → 10, 0 meV → 0
    thermo_normalized = min(10, max(0, -thermo_result.delta_g / 50))

    # Combine scores
    import math
    combined = (
        lit_weight * math.log1p(lit_score) +  # log(1+x) to dampen outliers
        thermo_weight * thermo_normalized
    )

    template.metadata["combined_score"] = combined

    return combined


def rank_templates_combined(
    templates: List["RecipeTemplate"],
    synthesis_dataset: "SynthesisDataset",
    arrows_integration: ARROWSIntegration,
    temperature: float,
    lit_weight: float = 0.5,
    thermo_weight: float = 0.5,
    allowed_byproducts: Optional[List[str]] = None,
) -> List["RecipeTemplate"]:
    """Rank templates by combined literature + thermodynamic score.

    Args:
        templates: List of RecipeTemplates to rank
        synthesis_dataset: SynthesisDataset for literature scoring
        arrows_integration: Initialized ARROWSIntegration
        temperature: Temperature in K
        lit_weight: Weight for literature score
        thermo_weight: Weight for thermodynamic score
        allowed_byproducts: Byproducts to consider

    Returns:
        Templates sorted by combined score (highest first)
    """
    for template in templates:
        score_template_combined(
            template,
            synthesis_dataset,
            arrows_integration,
            temperature,
            lit_weight,
            thermo_weight,
            allowed_byproducts,
        )

    return sorted(
        templates,
        key=lambda t: t.metadata.get("combined_score", float('-inf')),
        reverse=True,
    )
