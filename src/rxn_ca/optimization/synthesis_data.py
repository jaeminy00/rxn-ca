"""Utilities for working with text-mined synthesis datasets.

This module provides tools to query precursor frequencies and filter/rank
recipe templates based on what has been observed in the literature.

Data source: Kononova et al. "Text-mined dataset of inorganic materials
synthesis recipes" Scientific Data (2019)
https://doi.org/10.6084/m9.figshare.9722159.v3
"""

import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from pymatgen.core import Composition


@dataclass
class SynthesisRecord:
    """A single synthesis record from the text-mined dataset."""
    target_formula: str
    precursor_formulas: List[str]
    reaction_string: str
    doi: Optional[str] = None
    temperature: Optional[float] = None
    impurity_phases: List[str] = field(default_factory=list)  # NEW: what went wrong
    is_phase_pure: bool = True  # NEW: did synthesis succeed without impurities?

    @classmethod
    def from_dict(cls, data: dict) -> "SynthesisRecord":
        """Parse a record from either the 2019 or 2025 dataset format."""
        # Handle target - can be dict (2019) or list (2025)
        target_data = data.get("target", {})
        if isinstance(target_data, list):
            # 2025 LLM dataset format
            target_formula = target_data[0].get("material_formula", "") if target_data else ""
        else:
            # 2019 dataset format
            target_formula = target_data.get("material_formula", "")

        # Extract precursors
        precursors = []
        for p in data.get("precursors", []):
            formula = p.get("material_formula", "")
            if formula:
                precursors.append(formula)

        # Extract impurity phases (2025 dataset only)
        impurity_phases = []
        for imp in data.get("impurity_phase", []):
            formula = imp.get("material_formula", "")
            if formula:
                impurity_phases.append(formula)

        is_phase_pure = len(impurity_phases) == 0

        # Extract reaction string - different formats
        reaction_string = data.get("reaction_string", "")
        if not reaction_string:
            # 2025 format: target_reaction is [[target, {left, right}, null, string]]
            target_rxn = data.get("target_reaction", [])
            if target_rxn and len(target_rxn) > 0 and len(target_rxn[0]) > 3:
                reaction_string = target_rxn[0][3] or ""

        # Extract temperature from operations/conditions
        temp = None
        # Try 2019 format first
        for op in data.get("operations", []):
            conditions = op.get("conditions") or {}
            temps = conditions.get("heating_temperature", [])
            if temps:
                for t in temps:
                    if isinstance(t, dict) and "values" in t:
                        vals = t["values"]
                        if vals:
                            temp = vals[0]
                            break
                    elif isinstance(t, (int, float)):
                        temp = t
                        break
                if temp:
                    break

        # Try 2025 format (conditions_forDOI)
        if temp is None:
            for cond in data.get("conditions_forDOI", []):
                temp_vals = cond.get("temp_values", [])
                if temp_vals:
                    for tv in temp_vals:
                        if isinstance(tv, dict) and "values" in tv:
                            vals = tv["values"]
                            if vals:
                                temp = vals[0]
                                break
                    if temp:
                        break

        # Get DOI - different key names
        doi = data.get("doi") or data.get("DOI")

        return cls(
            target_formula=target_formula,
            precursor_formulas=precursors,
            reaction_string=reaction_string,
            doi=doi,
            temperature=temp,
            impurity_phases=impurity_phases,
            is_phase_pure=is_phase_pure,
        )


class SynthesisDataset:
    """Query interface for the text-mined synthesis dataset."""

    def __init__(self, records: List[SynthesisRecord]):
        """Initialize with parsed records."""
        self.records = records
        self._build_indices()

    @classmethod
    def from_json_file(cls, path: str) -> "SynthesisDataset":
        """Load dataset from JSON file (supports .json and .json.gz)."""
        path = Path(path)

        # Handle gzipped files
        if path.suffix == ".gz" or str(path).endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r") as f:
                data = json.load(f)

        records = []
        for item in data:
            try:
                record = SynthesisRecord.from_dict(item)
                if record.target_formula and record.precursor_formulas:
                    records.append(record)
            except Exception:
                continue

        return cls(records)

    @classmethod
    def from_multiple_files(cls, paths: List[str]) -> "SynthesisDataset":
        """Load and merge multiple dataset files."""
        all_records = []
        for path in paths:
            dataset = cls.from_json_file(path)
            all_records.extend(dataset.records)
        return cls(all_records)

    def _build_indices(self):
        """Build lookup indices for fast querying."""
        # Precursor frequency across all syntheses
        self.precursor_counts = Counter()

        # Precursor frequency per element (which precursors are used to source element X)
        self.element_precursor_counts: Dict[str, Counter] = defaultdict(Counter)

        # Precursor pair co-occurrence
        self.pair_counts = Counter()

        # Target -> precursors mapping
        self.target_to_precursors: Dict[str, List[List[str]]] = defaultdict(list)

        # NEW: Impurity tracking
        self.impurity_counts = Counter()  # How often each phase appears as impurity
        self.precursor_to_impurities: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.target_purity_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pure": 0, "impure": 0})

        # Counts for phase-pure vs phase-impure syntheses
        self.phase_pure_count = 0
        self.phase_impure_count = 0

        for record in self.records:
            precursors = record.precursor_formulas
            target = record.target_formula

            # Count precursors
            for p in precursors:
                self.precursor_counts[p] += 1

                # Count which elements this precursor provides
                try:
                    comp = Composition(p)
                    for el in comp.elements:
                        self.element_precursor_counts[str(el)][p] += 1
                except Exception:
                    continue

            # Count pairs
            for i, p1 in enumerate(precursors):
                for p2 in precursors[i+1:]:
                    pair = tuple(sorted([p1, p2]))
                    self.pair_counts[pair] += 1

            # Map target to precursors
            self.target_to_precursors[target].append(precursors)

            # Track impurities
            if record.is_phase_pure:
                self.phase_pure_count += 1
                self.target_purity_stats[target]["pure"] += 1
            else:
                self.phase_impure_count += 1
                self.target_purity_stats[target]["impure"] += 1

                # Track which impurities form from which precursor sets
                prec_key = tuple(sorted(precursors))
                for imp in record.impurity_phases:
                    self.impurity_counts[imp] += 1
                    self.precursor_to_impurities[prec_key][imp] += 1

    def get_precursor_frequency(self, precursor: str) -> int:
        """Get how many times a precursor appears in the dataset."""
        return self.precursor_counts.get(precursor, 0)

    def get_top_precursors(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get the N most common precursors."""
        return self.precursor_counts.most_common(n)

    def get_precursors_for_element(
        self,
        element: str,
        n: int = 10
    ) -> List[Tuple[str, int]]:
        """Get the most common precursors used to source a specific element."""
        return self.element_precursor_counts[element].most_common(n)

    def get_pair_frequency(self, p1: str, p2: str) -> int:
        """Get how often two precursors are used together."""
        pair = tuple(sorted([p1, p2]))
        return self.pair_counts.get(pair, 0)

    def get_precursors_for_target(self, target: str) -> List[List[str]]:
        """Get all precursor sets that have been used for a target."""
        return self.target_to_precursors.get(target, [])

    def score_precursor_set(
        self,
        precursors: List[str],
        method: str = "frequency"
    ) -> float:
        """Score a precursor set based on literature frequency.

        Args:
            precursors: List of precursor formulas
            method: Scoring method:
                - "frequency": Product of individual frequencies
                - "pair": Sum of pair co-occurrence frequencies
                - "geometric": Geometric mean of frequencies

        Returns:
            Score (higher = more commonly used in literature)
        """
        if method == "frequency":
            score = 1.0
            for p in precursors:
                freq = self.precursor_counts.get(p, 0)
                score *= (freq + 1)  # +1 to avoid zero
            return score

        elif method == "pair":
            score = 0
            for i, p1 in enumerate(precursors):
                for p2 in precursors[i+1:]:
                    pair = tuple(sorted([p1, p2]))
                    score += self.pair_counts.get(pair, 0)
            return score

        elif method == "geometric":
            import math
            log_score = 0
            for p in precursors:
                freq = self.precursor_counts.get(p, 0)
                log_score += math.log(freq + 1)
            return math.exp(log_score / len(precursors))

        else:
            raise ValueError(f"Unknown scoring method: {method}")

    def filter_by_frequency(
        self,
        precursors: List[str],
        min_frequency: int = 1,
    ) -> List[str]:
        """Filter precursors to only those observed in the dataset."""
        return [p for p in precursors if self.precursor_counts.get(p, 0) >= min_frequency]

    # =========================================================================
    # Impurity-related queries (requires 2025 LLM dataset)
    # =========================================================================

    def get_common_impurities(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get the most common impurity phases across all syntheses."""
        return self.impurity_counts.most_common(n)

    def get_purity_rate(self, target: Optional[str] = None) -> float:
        """Get the fraction of phase-pure syntheses.

        Args:
            target: If provided, get purity rate for specific target only

        Returns:
            Fraction of syntheses that were phase-pure (0.0 to 1.0)
        """
        if target:
            stats = self.target_purity_stats.get(target, {"pure": 0, "impure": 0})
            total = stats["pure"] + stats["impure"]
            return stats["pure"] / total if total > 0 else 0.0
        else:
            total = self.phase_pure_count + self.phase_impure_count
            return self.phase_pure_count / total if total > 0 else 0.0

    def get_impurities_for_precursors(
        self,
        precursors: List[str]
    ) -> List[Tuple[str, int]]:
        """Get impurities that have formed from a specific precursor set.

        Args:
            precursors: List of precursor formulas

        Returns:
            List of (impurity_phase, count) tuples
        """
        prec_key = tuple(sorted(precursors))
        return self.precursor_to_impurities[prec_key].most_common()

    def get_phase_pure_records(self) -> List[SynthesisRecord]:
        """Get all records where synthesis was phase-pure."""
        return [r for r in self.records if r.is_phase_pure]

    def get_phase_impure_records(self) -> List[SynthesisRecord]:
        """Get all records where impurities formed."""
        return [r for r in self.records if not r.is_phase_pure]

    def get_records_for_target(
        self,
        target: str,
        phase_pure_only: bool = False
    ) -> List[SynthesisRecord]:
        """Get all synthesis records for a specific target.

        Args:
            target: Target phase formula
            phase_pure_only: If True, only return phase-pure syntheses

        Returns:
            List of SynthesisRecord objects
        """
        records = [r for r in self.records if r.target_formula == target]
        if phase_pure_only:
            records = [r for r in records if r.is_phase_pure]
        return records

    def suggest_precursors_for_target(
        self,
        target: str,
        available_phases: List[str],
        n_suggestions: int = 5,
    ) -> List[Tuple[List[str], float]]:
        """Suggest precursor sets for a target based on literature.

        First checks if the exact target has been synthesized before.
        If not, looks for similar targets (same elements).

        Args:
            target: Target phase formula
            available_phases: Phases available in the system
            n_suggestions: Number of suggestions to return

        Returns:
            List of (precursor_set, score) tuples
        """
        available_set = set(available_phases)
        suggestions = []

        # Check exact matches
        exact_precursors = self.target_to_precursors.get(target, [])
        for prec_set in exact_precursors:
            # Filter to available phases
            filtered = [p for p in prec_set if p in available_set]
            if len(filtered) >= 2:
                score = self.score_precursor_set(filtered, method="pair")
                suggestions.append((filtered, score + 1000))  # Bonus for exact match

        # Check similar targets (same elements)
        try:
            target_comp = Composition(target)
            target_elements = {str(el) for el in target_comp.elements}
        except Exception:
            target_elements = set()

        if target_elements:
            for t, prec_sets in self.target_to_precursors.items():
                if t == target:
                    continue
                try:
                    t_comp = Composition(t)
                    t_elements = {str(el) for el in t_comp.elements}
                    if t_elements == target_elements:
                        for prec_set in prec_sets:
                            filtered = [p for p in prec_set if p in available_set]
                            if len(filtered) >= 2:
                                score = self.score_precursor_set(filtered, method="pair")
                                suggestions.append((filtered, score))
                except Exception:
                    continue

        # Deduplicate and sort
        seen = set()
        unique_suggestions = []
        for prec_set, score in sorted(suggestions, key=lambda x: -x[1]):
            key = tuple(sorted(prec_set))
            if key not in seen:
                seen.add(key)
                unique_suggestions.append((list(prec_set), score))

        return unique_suggestions[:n_suggestions]


def get_practical_precursors(
    dataset: SynthesisDataset,
    elements: List[str],
    min_frequency: int = 5,
    max_per_element: int = 5,
) -> Dict[str, List[str]]:
    """Get practical precursors for each element based on literature frequency.

    Args:
        dataset: SynthesisDataset instance
        elements: List of elements to find precursors for
        min_frequency: Minimum times a precursor must appear in literature
        max_per_element: Maximum precursors to return per element

    Returns:
        Dict mapping element -> list of practical precursor formulas
    """
    result = {}
    for el in elements:
        top_precursors = dataset.get_precursors_for_element(el, n=max_per_element * 2)
        # Filter by minimum frequency
        filtered = [(p, count) for p, count in top_precursors if count >= min_frequency]
        result[el] = [p for p, _ in filtered[:max_per_element]]
    return result


# Singleton for lazy loading
_dataset_instance: Optional[SynthesisDataset] = None
_dataset_path: Optional[str] = None


def load_synthesis_dataset(path: Optional[str] = None) -> SynthesisDataset:
    """Load the synthesis dataset (cached singleton).

    Args:
        path: Path to JSON file. If None, looks for default locations.

    Returns:
        SynthesisDataset instance
    """
    global _dataset_instance, _dataset_path

    if path is None:
        # Try default locations
        default_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "data" / "solid_state_synthesis.json",
            Path.home() / ".rxn_ca" / "solid_state_synthesis.json",
        ]
        for p in default_paths:
            if p.exists():
                path = str(p)
                break

    if path is None:
        raise FileNotFoundError(
            "Synthesis dataset not found. Download from: "
            "https://figshare.com/articles/dataset/solid-state_dataset_2019-06-27_upd_json/9722159"
        )

    if _dataset_instance is None or _dataset_path != path:
        _dataset_instance = SynthesisDataset.from_json_file(path)
        _dataset_path = path

    return _dataset_instance
