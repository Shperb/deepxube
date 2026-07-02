from deepxube.domains.lean.types import LeanState, LeanAction, DEAD
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend, TacticOutcome
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _domain():
    tree = {
        "u@c:F.lean:t": {
            "__init__": "⊢ p → p",
            ("intro h",): TacticOutcome(pp="h : p ⊢ p", done=False),
            ("intro h", "exact h"): TacticOutcome(pp="no goals", done=True),
            ("intro h", "bad"): TacticOutcome(pp=None, done=False, error="unknown"),
        }
    }
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=2)])
    return LeanDomain(FakeLeanBackend(tree), FakeTacticGenerator({}), corpus, k=2, model_name="m", seed=0)


def test_next_state_applies_tactic_and_extends_path():
    d = _domain()
    s = LeanState("u@c:F.lean:t", ("intro h",), "h : p ⊢ p", done=False)
    (nxt,), (tc,) = d.next_state([s], [LeanAction("exact h")])
    assert tc == 1.0 and nxt.done and nxt.pp == "no goals"
    assert nxt.tactic_path == ("intro h", "exact h")


def test_invalid_tactic_maps_to_dead_with_unit_cost():
    d = _domain()
    s = LeanState("u@c:F.lean:t", ("intro h",), "h : p ⊢ p", done=False)
    (nxt,), (tc,) = d.next_state([s], [LeanAction("bad")])
    assert nxt is DEAD and tc == 1.0


def test_batch_routes_by_theorem_and_preserves_order():
    d = _domain()
    s0 = LeanState("u@c:F.lean:t", (), "⊢ p → p", done=False)
    s1 = LeanState("u@c:F.lean:t", ("intro h",), "h : p ⊢ p", done=False)
    states_next, tcs = d.next_state([s1, s0], [LeanAction("exact h"), LeanAction("intro h")])
    assert states_next[0].pp == "no goals" and states_next[1].pp == "h : p ⊢ p"
    assert tcs == [1.0, 1.0]
