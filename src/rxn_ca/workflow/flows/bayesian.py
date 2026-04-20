"""Bayesian Optimization flow Maker for ReactCA synthesis."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from jobflow import Flow, Maker

from rxn_ca.optimization import SearchSpace
from ..jobs.core import setup_reaction_library
from ..jobs.bayesian import init_bo_campaign, bo_trial_step


@dataclass
class BOFlowMaker(Maker):
    """Maker for Bayesian Optimization flows over ReactCA simulations.

    Stores optimization configuration as dataclass fields so the same
    Maker instance can be reused for multiple chemical systems without
    re-specifying all parameters. Can be serialized with monty and
    composed into larger workflow Makers.

    Wires together three job stages:
        setup_reaction_library  →  init_bo_campaign  →  bo_trial_step[0]
                                                               ↓ Response.addition
                                                        bo_trial_step[1] → ...

    Attributes:
        name: Name prefix for generated Flows. Full flow name is
            "{name}_{target_phase}_{chemical_system}".
        n_initial: Number of random exploration trials before BO guidance.
        n_iterations: Number of BO-guided trials after the initial phase.
        scorer_type: How to score each simulation result.
            "final" — yield of target phase at the end of the simulation.
            "maximum" — peak yield of target phase during the simulation.
        simulation_size: CA grid size (NxN cells).
        num_realizations: Number of independent simulation runs per trial.
            Results are averaged for the BO score.
        live_compress: Store full CA state snapshots at each compress_freq
            step instead of diffs. Strongly recommended.
        compress_freq: Interval (in simulation steps) between snapshots
            when live_compress is True.
        metastability_cutoff: Energy above the convex hull (eV/atom) below
            which phases are included.
        exclude_theoretical: If True, exclude phases without experimental
            observations in the Materials Project database.

    Example — reuse the same maker for different systems:
        maker = BOFlowMaker(n_initial=5, n_iterations=15, simulation_size=10)
        flow_li = maker.make("Li-Si-O-C", "Li4SiO4", search_space_li, output_dir_li)
        flow_ba = maker.make("Ba-Ti-O",   "BaTiO3",  search_space_ba, output_dir_ba)

    Example — compose into a larger Maker:
        @dataclass
        class MySweepMaker(Maker):
            name: str = "sweep"
            bo_maker: BOFlowMaker = field(default_factory=BOFlowMaker)

            def make(self, systems: list, ...) -> Flow:
                flows = [self.bo_maker.make(sys, ...) for sys in systems]
                return Flow(flows, name=self.name)
    """

    name: str = "bo_flow"
    n_initial: int = 5
    n_iterations: int = 15
    scorer_type: str = "final"
    simulation_size: int = 10
    num_realizations: int = 3
    live_compress: bool = True
    compress_freq: int = 50
    metastability_cutoff: float = 0.1
    exclude_theoretical: bool = True

    def make(
        self,
        chemical_system: str,
        target_phase: str,
        search_space: SearchSpace,
        output_dir: str,
        fixed_precursors: Optional[Dict[str, float]] = None,
        ensure_phases: Optional[List[str]] = None,
        **library_kwargs,
    ) -> Flow:
        """Build a Bayesian Optimization Flow for the given chemical system.

        Args:
            chemical_system: Element system string, e.g. "Li-Si-O-C".
            target_phase: Formula of the target product, e.g. "Li4SiO4".
            search_space: Configured SearchSpace. Must contain a 'hold_temp'
                parameter; precursor slots (if any) are derived automatically.
            output_dir: Shared filesystem path written to by every trial job
                (history.csv, best_result.json, simulations/). Must be
                accessible from all HPC worker nodes.
            fixed_precursors: Formula → molar amount map. When provided,
                precursor selection is not optimized (only the thermal profile
                is). Mutually exclusive with precursor slots in search_space.
            ensure_phases: Phases that must be present in the reaction library.
                Defaults to target_phase + all precursor candidates.
            **library_kwargs: Forwarded to setup_reaction_library
                (e.g. thermo_types=["R2SCAN"]).

        Returns:
            Flow containing setup, init, and first trial jobs. Subsequent trial
            jobs are added dynamically at runtime via Response(addition=...).

        Raises:
            ValueError: If search_space has no 'hold_temp' parameter.
            ValueError: If both fixed_precursors and precursor slots are given.
        """
        flow_name = f"{self.name}_{target_phase}_{chemical_system}"
        output_dir = str(Path(output_dir).expanduser().resolve())
        total_iterations = self.n_initial + self.n_iterations

        # --- Derive temperatures from search space hold_temp parameter ---
        hold_temp_param = search_space.get_parameter("hold_temp")
        if hold_temp_param is None:
            raise ValueError(
                "search_space must contain a 'hold_temp' parameter. "
                "Add one with search_space.add_temperature_range(...)."
            )
        temperatures = sorted(
            set(int(t) for t in np.arange(300, hold_temp_param.high + 1, 100))
        )

        # --- Derive precursor slot names from search space ---
        precursor_slot_names = [p.name for p in search_space.precursor_parameters]

        if fixed_precursors is not None and precursor_slot_names:
            raise ValueError(
                "Provide either fixed_precursors or precursor slots in search_space, not both."
            )

        # --- Build ensure_phases if not explicitly provided ---
        if ensure_phases is None:
            ensure_phases = [target_phase]
            if fixed_precursors:
                ensure_phases.extend(fixed_precursors.keys())
            else:
                for param in search_space.precursor_parameters:
                    ensure_phases.extend(param.candidates)
            ensure_phases = list(dict.fromkeys(ensure_phases))

        # --- Objective config passed to every trial job ---
        objective_config = {
            "target_phase": target_phase,
            "scorer_type": self.scorer_type,
            "simulation_size": self.simulation_size,
            "num_realizations": self.num_realizations,
            "live_compress": self.live_compress,
            "compress_freq": self.compress_freq,
            "target_name": "yield",
        }

        # --- Job 1: Build reaction library (runs once, shared across all trials) ---
        setup_job = setup_reaction_library(
            chemical_system=chemical_system,
            temperatures=temperatures,
            ensure_phases=ensure_phases,
            metastability_cutoff=self.metastability_cutoff,
            exclude_theoretical=self.exclude_theoretical,
            save_to_file=True,
            entry_kwargs=library_kwargs.get("entry_kwargs", {}),
        )
        setup_job.name = f"setup_{chemical_system}"

        # --- Job 2: Initialize BayBE Campaign ---
        init_job = init_bo_campaign(
            search_space_config=search_space.as_dict(),
            n_initial=self.n_initial,
            n_iterations=self.n_iterations,
        )
        init_job.name = "init_bo_campaign"

        # --- Job 3: First BO trial ---
        # campaign_json and reaction_library_data are job output references —
        # jobflow resolves them at runtime after the upstream jobs complete.
        first_trial = bo_trial_step(
            iteration=0,
            total_iterations=total_iterations,
            campaign_json=init_job.output["campaign_json"],
            reaction_library_data=setup_job.output,
            precursor_slot_names=precursor_slot_names,
            fixed_precursors=fixed_precursors,
            objective_config=objective_config,
            output_dir=output_dir,
        )
        first_trial.name = "bo_trial_000"

        return Flow(
            [setup_job, init_job, first_trial],
            name=flow_name,
        )
