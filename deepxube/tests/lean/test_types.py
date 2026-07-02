import pickle
from deepxube.domains.lean.types import LeanState, LeanAction, LeanGoal, DEAD


def test_state_equality_and_hash_by_theorem_and_pp():
    a = LeanState("thm1", ("intro h",), "⊢ p → q", done=False)
    b = LeanState("thm1", ("intro h", "exact h"), "⊢ p → q", done=False)  # different path, same pp+thm
    c = LeanState("thm2", (), "⊢ p → q", done=False)
    assert a == b and hash(a) == hash(b)
    assert a != c


def test_state_is_picklable():
    s = LeanState("t", ("rfl",), "no goals", done=True)
    assert pickle.loads(pickle.dumps(s)) == s


def test_dead_sentinel_is_singleton_and_distinct():
    assert DEAD is LeanState.dead()
    assert DEAD != LeanState("t", (), "⊢ p", done=False)


def test_action_repr_is_tactic_string():
    act = LeanAction("simp")
    assert repr(act) == "simp"
    assert LeanAction("simp") == LeanAction("simp") and hash(LeanAction("simp")) == hash(LeanAction("simp"))


def test_goal_empty_vs_relabeled():
    assert LeanGoal(None).is_empty()
    assert not LeanGoal("⊢ p").is_empty()
