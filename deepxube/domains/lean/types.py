from typing import Optional, Tuple
from deepxube.base.domain import State, Action, Goal

_DEAD_ID = "__DEAD__"


class LeanState(State):
    """ A Lean proof state. Serializable so it can cross process boundaries and be replayed.

    :param theorem_id: unique id of the theorem whose Dojo produced this state
    :param tactic_path: sequence of tactic strings from the theorem's initial state to here (for replay)
    :param pp: LeanDojo pretty-printed tactic state (content identity)
    :param done: True iff this state is a completed proof (ProofFinished)
    """
    __slots__ = ["theorem_id", "tactic_path", "pp", "done"]

    def __init__(self, theorem_id: str, tactic_path: Tuple[str, ...], pp: str, done: bool):
        self.theorem_id: str = theorem_id
        self.tactic_path: Tuple[str, ...] = tuple(tactic_path)
        self.pp: str = pp
        self.done: bool = done

    @staticmethod
    def dead() -> "LeanState":
        return DEAD

    def is_dead(self) -> bool:
        return self.theorem_id == _DEAD_ID

    def __hash__(self) -> int:
        return hash((self.theorem_id, self.pp))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LeanState):
            return (self.theorem_id == other.theorem_id) and (self.pp == other.pp)
        return NotImplemented

    def __repr__(self) -> str:
        return f"LeanState(thm={self.theorem_id}, done={self.done}, pp={self.pp[:40]!r})"


DEAD: LeanState = LeanState(_DEAD_ID, (), "__DEAD__", done=False)


class LeanAction(Action):
    """ A tactic to apply. """
    __slots__ = ["tactic"]

    def __init__(self, tactic: str):
        self.tactic: str = tactic

    def __hash__(self) -> int:
        return hash(self.tactic)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LeanAction):
            return self.tactic == other.tactic
        return NotImplemented

    def __repr__(self) -> str:
        return self.tactic


class LeanGoal(Goal):
    """ Goal-conditioned target tactic state. target_pp None => empty state (real proof objective). """
    __slots__ = ["target_pp"]

    def __init__(self, target_pp: Optional[str]):
        self.target_pp: Optional[str] = target_pp

    def is_empty(self) -> bool:
        return self.target_pp is None

    def __repr__(self) -> str:
        return "LeanGoal(∅)" if self.target_pp is None else f"LeanGoal({self.target_pp[:40]!r})"
