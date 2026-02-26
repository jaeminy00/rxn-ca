from pylattica.core import SimulationState, SimulationResult


class ReactionResult(SimulationResult):
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    # Inherits from_dict from SimulationResult - it uses cls() so creates ReactionResult

    def __init__(
        self,
        starting_state: SimulationState,
        compress_freq: int = 1,
        max_history: int = None,
        live_compress: bool = False,
    ):
        """Initializes a ReactionResult.

        Args:
            starting_state: The initial simulation state.
            compress_freq: Interval for storing frames when live_compress is True.
            max_history: Max diffs to keep before checkpointing (None = unlimited).
            live_compress: If True, store full state snapshots at compress_freq
                intervals instead of diffs. Avoids slow reconstruction in analysis.
        """
        super().__init__(
            starting_state,
            compress_freq=compress_freq,
            max_history=max_history,
            live_compress=live_compress,
        )
    