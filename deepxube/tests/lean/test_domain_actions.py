from deepxube.domains.lean.types import LeanState, LeanAction, DEAD
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend, TacticOutcome
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef

TID = "u@c:F.lean:t"


def _domain(gen_table, tree, k=2, resample_attempts=2):
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    be = FakeLeanBackend({TID: {"__init__": "⊢ p", **tree}})
    return LeanDomain(be, FakeTacticGenerator(gen_table), corpus, k=k, model_name="m", seed=0,
                      resample_attempts=resample_attempts)


def test_get_state_actions_returns_valid_top_k():
    # generator proposes 3 tactics; 'exact h' is rejected by Lean -> only valid ones returned
    d = _domain(
        gen_table={"⊢ p": ["intro h", "simp", "exact h"]},
        tree={
            ("intro h",): TacticOutcome(pp="h : p ⊢ p", done=False),
            ("simp",): TacticOutcome(pp="⊢ q", done=False),
            ("exact h",): TacticOutcome(pp=None, done=False, error="unknown identifier"),
        },
    )
    s = LeanState(TID, (), "⊢ p", done=False)
    assert d.get_state_actions([s]) == [[LeanAction("intro h"), LeanAction("simp")]]


def test_done_and_dead_states_have_no_actions():
    d = _domain(gen_table={"⊢ p": ["intro h"]}, tree={("intro h",): TacticOutcome(pp="x", done=False)})
    done = LeanState(TID, ("rfl",), "no goals", done=True)
    assert d.get_state_actions([done, DEAD]) == [[], []]


def test_resamples_when_top_k_all_invalid():
    # first 2 candidates invalid; growing the beam surfaces valid ones
    d = _domain(
        gen_table={"⊢ p": ["bad1", "bad2", "good", "good2"]},
        tree={
            ("bad1",): TacticOutcome(pp=None, done=False, error="e"),
            ("bad2",): TacticOutcome(pp=None, done=False, error="e"),
            ("good",): TacticOutcome(pp="s1", done=False),
            ("good2",): TacticOutcome(pp="s2", done=False),
        },
    )
    s = LeanState(TID, (), "⊢ p", done=False)
    assert [a.tactic for a in d.get_state_actions([s])[0]] == ["good", "good2"]


def test_stuck_state_returns_no_actions_after_resampling():
    # all candidates invalid, even after resampling -> genuinely stuck (childless, no DEAD sink)
    d = _domain(
        gen_table={"⊢ p": ["bad1", "bad2"]},
        tree={("bad1",): TacticOutcome(pp=None, done=False, error="e"),
              ("bad2",): TacticOutcome(pp=None, done=False, error="e")},
    )
    s = LeanState(TID, (), "⊢ p", done=False)
    assert d.get_state_actions([s]) == [[]]
