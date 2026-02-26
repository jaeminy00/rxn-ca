"""Output schemas for rxn-ca workflow jobs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReactionLibraryData:
    """Output from the setup_reaction_library job.

    Contains the serialized phase set and reaction library, plus metadata.
    The reaction library can optionally be saved to a file for large systems.
    """

    phase_set_dict: Dict[str, Any]
    reaction_library_dict: Dict[str, Any]
    chemical_system: str
    temperatures: List[float]
    phases_available: List[str]

    # If the library was saved to a file, this is the path
    reaction_library_path: Optional[str] = None


@dataclass
class SimulationOutput:
    """Output from an rxn-ca simulation job.

    Contains analyzed results for quick access, plus file references
    for retrieving full simulation data when needed.

    The phase_set enables conversions between molar/mass/volume amounts:
        phase_set = SolidPhaseSet.from_dict(output.phase_set_dict)
        volume = moles * phase_set.get_vol(phase)
        mass = moles * Composition(phase).weight
    """

    # Analyzed results for quick access
    final_molar_amounts: Dict[str, float]
    molar_amounts_trajectory: Dict[str, List[float]]
    temperature_trajectory: List[float]
    step_indices: List[int]

    # Phase set for unit conversions (relatively small, safe to serialize)
    phase_set_dict: Dict[str, Any]

    # File references for full data (avoids serializing large objects)
    reaction_library_path: Optional[str] = None
    result_doc_path: Optional[str] = None

    # Recipe and metadata
    recipe_dict: Dict[str, Any] = field(default_factory=dict)
    chemical_system: Optional[str] = None
    num_realizations: int = 1
    metadata: Optional[Dict[str, Any]] = None
