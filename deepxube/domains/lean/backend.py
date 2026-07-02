from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class TacticOutcome:
    """ Result of applying one tactic. Exactly one of (pp set) or (error set) is meaningful. """
    pp: Optional[str]
    done: bool
    error: Optional[str] = None


@runtime_checkable
class LeanBackend(Protocol):
    """ Minimal Lean interaction surface the domain depends on. Implementations manage sessions internally. """

    def initial_pp(self, theorem_id: str) -> str:
        """ Pretty-printed initial tactic state for the theorem. """
        ...

    def run(self, theorem_id: str, tactic_path: Tuple[str, ...], tactic: str) -> TacticOutcome:
        """ Reach the state described by tactic_path (replaying if needed), then apply `tactic`. """
        ...


class FakeLeanBackend:
    """ In-memory backend for tests. `tree[theorem_id]` maps '__init__' -> pp and tactic-path tuples -> TacticOutcome. """

    def __init__(self, tree: Dict[str, Dict[object, object]]):
        self._tree = tree

    def initial_pp(self, theorem_id: str) -> str:
        return self._tree[theorem_id]["__init__"]  # type: ignore[return-value]

    def run(self, theorem_id: str, tactic_path: Tuple[str, ...], tactic: str) -> TacticOutcome:
        key = tuple(tactic_path) + (tactic,)
        outcome = self._tree[theorem_id].get(key)
        if outcome is None:
            return TacticOutcome(pp=None, done=False, error=f"no fake outcome for {key}")
        assert isinstance(outcome, TacticOutcome)
        return outcome
