from deepxube.domains.lean.types import LeanGoal
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _domain():
    refs = [TheoremRef("u", "c", "F.lean", f"t{n}") for n in [1, 5]]
    corpus = Corpus([CorpusEntry(refs[0], proof_len=1), CorpusEntry(refs[1], proof_len=5)])
    tree = {r.theorem_id: {"__init__": f"⊢ goal_{r.theorem_name}"} for r in refs}
    return LeanDomain(FakeLeanBackend(tree), FakeTacticGenerator({}), corpus, k=2, model_name="m", seed=7)


def test_sample_problem_instances_returns_states_and_empty_goals():
    d = _domain()
    states, goals = d.sample_problem_instances([1, 5])
    assert len(states) == len(goals) == 2
    assert all(isinstance(g, LeanGoal) and g.is_empty() for g in goals)
    assert all(s.tactic_path == () and not s.done for s in states)
    assert states[0].pp.startswith("⊢ goal_")


def test_sample_problem_instances_is_deterministic():
    d1, d2 = _domain(), _domain()
    s1, _ = d1.sample_problem_instances([1, 5])
    s2, _ = d2.sample_problem_instances([1, 5])
    assert [s.theorem_id for s in s1] == [s.theorem_id for s in s2]
