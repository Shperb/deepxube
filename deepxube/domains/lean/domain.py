import json
import os
from typing import Dict, List, Optional, Tuple

from deepxube.base.domain import ActsEnum, GoalSampleableFromState, StringToAct
from deepxube.base.factory import DelimParser
from deepxube.domains.lean.types import LeanState, LeanAction, LeanGoal
from deepxube.domains.lean.backend import LeanBackend
from deepxube.domains.lean.policy import TacticGenerator, CachedTacticGenerator
from deepxube.domains.lean.corpus import Corpus
from deepxube.factories.domain_factory import domain_factory
from deepxube.utils.timing_utils import Times


class LeanDomain(ActsEnum[LeanState, LeanAction, LeanGoal],
                 GoalSampleableFromState[LeanState, LeanAction, LeanGoal],
                 StringToAct[LeanState, LeanAction, LeanGoal]):
    """ Lean 4 theorem-proving domain. Actions are top-k ReProver tactics; goal-conditioned via HER. """

    def __init__(self, backend: LeanBackend, generator: TacticGenerator, corpus: Corpus,
                 k: int, model_name: str, seed: int, resample_attempts: int = 2):
        super().__init__()
        self.backend: LeanBackend = backend
        self.generator: CachedTacticGenerator = CachedTacticGenerator(generator)
        self.corpus: Corpus = corpus
        self.k: int = k
        self.model_name: str = model_name
        self.seed: int = seed
        # If fewer than k candidate tactics validate, grow the beam deterministically and retry
        # up to this many times (candidates cannot hurt: they only enlarge the successor set).
        self.resample_attempts: int = resample_attempts
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

    def _valid_tactics(self, state: LeanState, candidates: List[str]) -> List[str]:
        """ Keep only candidates that Lean accepts (validated via the backend; results are cached
        so next_state reuses them). Preserves candidate order. """
        valid: List[str] = []
        for tactic in candidates:
            outcome = self.backend.run(state.theorem_id, state.tactic_path, tactic)
            if outcome.error is None and outcome.pp is not None:
                valid.append(tactic)
        return valid

    def get_state_actions(self, states: List[LeanState]) -> List[List[LeanAction]]:
        # terminal/dead states expand to nothing
        need_idx: List[int] = [i for i, s in enumerate(states) if not (s.done or s.is_dead())]
        actions_l: List[List[LeanAction]] = [[] for _ in states]
        if not need_idx:
            return actions_l

        # batched initial generation, then per-state validation (+ deterministic resampling)
        cand_l: List[List[str]] = self.generator.top_k([states[i].pp for i in need_idx], self.k)
        for local_i, state_i in enumerate(need_idx):
            state = states[state_i]
            candidates: List[str] = cand_l[local_i]
            valid: List[str] = self._valid_tactics(state, candidates)

            k_try: int = self.k
            attempts: int = 0
            # grow the beam only while the generator is not exhausted and we still lack k valid ones
            while len(valid) < self.k and len(candidates) >= k_try and attempts < self.resample_attempts:
                k_try *= 2
                candidates = self.generator.top_k([state.pp], k_try)[0]
                valid = self._valid_tactics(state, candidates)
                attempts += 1

            actions_l[state_i] = [LeanAction(t) for t in valid[:self.k]]
        return actions_l

    def next_state(self, states: List[LeanState],
                   actions: List[LeanAction]) -> Tuple[List[LeanState], List[float]]:
        states_next: List[LeanState] = []
        for state, action in zip(states, actions, strict=True):
            if state.done or state.is_dead():
                # terminal/dead states should not be expanded; guard defensively
                states_next.append(LeanState.dead())
                continue
            outcome = self.backend.run(state.theorem_id, state.tactic_path, action.tactic)
            if outcome.error is not None or outcome.pp is None:
                states_next.append(LeanState.dead())
            else:
                states_next.append(LeanState(
                    theorem_id=state.theorem_id,
                    tactic_path=state.tactic_path + (action.tactic,),
                    pp=outcome.pp,
                    done=outcome.done,
                ))
        return states_next, [1.0] * len(states_next)

    def sample_problem_instances(self, num_steps_l: List[int],
                                 times: Optional[Times] = None) -> Tuple[List[LeanState], List[LeanGoal]]:
        # deterministic curriculum sampling; advance the seed per call for variety across iterations
        call_seed: int = hash((self.seed, self._sample_calls)) & 0x7FFFFFFF
        self._sample_calls += 1
        entries = self.corpus.sample_theorems(num_steps_l, seed=call_seed)

        states: List[LeanState] = []
        for entry in entries:
            tid = entry.ref.theorem_id
            states.append(LeanState(theorem_id=tid, tactic_path=(), pp=self.backend.initial_pp(tid), done=False))
        goals: List[LeanGoal] = [LeanGoal(None) for _ in states]
        return states, goals


@domain_factory.register_class("lean")
class LeanDomainCLI(LeanDomain):
    """ CLI-constructible LeanDomain: builds the real LeanDojo backend + ReProver generator + corpus from a JSON manifest.

    Manifest schema: {"theorems": [{"url","commit","file_path","theorem_name","proof_len"?}, ...]}
    Heavy deps (lean_dojo, transformers) are imported lazily inside __init__ / only loaded when actually used.
    """
    def __init__(self, k: int = 8, manifest: str = "lean_corpus.json",
                 model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
                 seed: int = 0, max_sessions: int = 8):
        from deepxube.domains.lean.dojo_backend import LeanDojoBackend, TheoremRef
        from deepxube.domains.lean.policy import ReProverGenerator
        from deepxube.domains.lean.corpus import Corpus, CorpusEntry

        assert os.path.exists(manifest), f"corpus manifest not found: {manifest}"
        with open(manifest, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries: List[CorpusEntry] = []
        refs: Dict[str, TheoremRef] = {}
        for item in data["theorems"]:
            ref = TheoremRef(item["url"], item["commit"], item["file_path"], item["theorem_name"])
            refs[ref.theorem_id] = ref
            entries.append(CorpusEntry(ref, proof_len=item.get("proof_len")))
        backend = LeanDojoBackend(refs, max_sessions=max_sessions)
        generator = ReProverGenerator(model_name=model_name)
        super().__init__(backend=backend, generator=generator, corpus=Corpus(entries),
                         k=k, model_name=model_name, seed=seed)


def build_lean_domain(k: int = 8, manifest: str = "lean_corpus.json",
                      model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
                      seed: int = 0, max_sessions: int = 8) -> "LeanDomainCLI":
    """ Convenience factory returning a CLI-constructed LeanDomain from a JSON manifest. """
    return LeanDomainCLI(k=k, manifest=manifest, model_name=model_name, seed=seed, max_sessions=max_sessions)


@domain_factory.register_parser("lean")
class LeanParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("k", "k", int, "number of top tactics per state")
        self.add_argument("seed", "seed", int, "random seed")

    @property
    def delim(self) -> str:
        return "_"
