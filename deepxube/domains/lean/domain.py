from typing import List, Optional, Tuple

from deepxube.base.domain import ActsEnum, GoalSampleableFromState, StringToAct
from deepxube.domains.lean.types import LeanState, LeanAction, LeanGoal
from deepxube.domains.lean.backend import LeanBackend
from deepxube.domains.lean.policy import TacticGenerator, CachedTacticGenerator
from deepxube.domains.lean.corpus import Corpus
from deepxube.utils.timing_utils import Times


class LeanDomain(ActsEnum[LeanState, LeanAction, LeanGoal],
                  GoalSampleableFromState[LeanState, LeanAction, LeanGoal],
                  StringToAct[LeanState, LeanAction, LeanGoal]):
    """ Lean 4 theorem-proving domain. Actions are top-k ReProver tactics; goal-conditioned via HER. """

    def __init__(self, backend: LeanBackend, generator: TacticGenerator, corpus: Corpus,
                 k: int, model_name: str, seed: int):
        super().__init__()
        self.backend: LeanBackend = backend
        self.generator: CachedTacticGenerator = CachedTacticGenerator(generator)
        self.corpus: Corpus = corpus
        self.k: int = k
        self.model_name: str = model_name
        self.seed: int = seed
        self._sample_calls: int = 0

    def is_solved(self, states: List[LeanState], goals: List[LeanGoal]) -> List[bool]:
        out: List[bool] = []
        for state, goal in zip(states, goals, strict=True):
            if goal.is_empty():
                out.append(state.done)
            else:
                out.append((not state.is_dead()) and state.pp == goal.target_pp)
        return out

    def sample_goal_from_state(self, states_start: Optional[List[LeanState]],
                                states_goal: List[LeanState]) -> List[LeanGoal]:
        # HER relabeling: the goal is exactly the reached tactic state
        return [LeanGoal(target_pp=s.pp) for s in states_goal]

    def string_to_action(self, act_str: str) -> Optional[LeanAction]:
        act_str = act_str.strip()
        return LeanAction(act_str) if act_str else None

    def string_to_action_help(self) -> str:
        return "Type a Lean tactic string (e.g. 'intro h', 'simp', 'exact h')."

    def __repr__(self) -> str:
        return f"LeanDomain(k={self.k}, model={self.model_name})"

    # --- Not yet implemented; filled in by later tasks ---
    def get_state_actions(self, states: List[LeanState]) -> List[List[LeanAction]]:
        raise NotImplementedError("implemented in Task 9")

    def next_state(self, states: List[LeanState], actions: List[LeanAction]) -> Tuple[List[LeanState], List[float]]:
        raise NotImplementedError("implemented in Task 10")

    def sample_problem_instances(self, num_steps_l: List[int],
                                  times: Optional[Times] = None) -> Tuple[List[LeanState], List[LeanGoal]]:
        raise NotImplementedError("implemented in Task 11")
