from deepxube.domains.lean.types import LeanState, LeanGoal
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _domain():
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    be = FakeLeanBackend({"u@c:F.lean:t": {"__init__": "⊢ p"}})
    gen = FakeTacticGenerator({})
    return LeanDomain(backend=be, generator=gen, corpus=corpus, k=2, model_name="m", seed=0)


def test_is_solved_empty_goal_uses_done_flag():
    d = _domain()
    s_done = LeanState("t", ("rfl",), "no goals", done=True)
    s_open = LeanState("t", (), "⊢ p", done=False)
    assert d.is_solved([s_done, s_open], [LeanGoal(None), LeanGoal(None)]) == [True, False]


def test_is_solved_relabeled_goal_matches_pp():
    d = _domain()
    s = LeanState("t", ("intro h",), "h : p ⊢ p", done=False)
    assert d.is_solved([s], [LeanGoal("h : p ⊢ p")]) == [True]
    assert d.is_solved([s], [LeanGoal("⊢ other")]) == [False]


def test_sample_goal_from_state_relabels_to_reached_pp():
    d = _domain()
    reached = LeanState("t", ("intro h",), "h : p ⊢ p", done=False)
    goals = d.sample_goal_from_state(None, [reached])
    assert goals[0].target_pp == "h : p ⊢ p"
