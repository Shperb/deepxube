from deepxube.domains.lean.types import LeanState, LeanAction, DEAD
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _domain(table):
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    be = FakeLeanBackend({"u@c:F.lean:t": {"__init__": "⊢ p"}})
    return LeanDomain(backend=be, generator=FakeTacticGenerator(table), corpus=corpus, k=2, model_name="m", seed=0)


def test_get_state_actions_returns_top_k_tactics():
    d = _domain({"⊢ p": ["intro h", "simp", "exact h"]})
    s = LeanState("t", (), "⊢ p", done=False)
    acts = d.get_state_actions([s])
    assert acts == [[LeanAction("intro h"), LeanAction("simp")]]


def test_done_and_dead_states_have_no_actions():
    d = _domain({"⊢ p": ["intro h"]})
    done = LeanState("t", ("rfl",), "no goals", done=True)
    assert d.get_state_actions([done, DEAD]) == [[], []]
