from deepxube.domains.lean.backend import TacticOutcome, FakeLeanBackend


def _tree():
    # theorem "t": intro -> "s1"; on s1: exact -> proof done; bad -> error
    return {
        "t": {
            "__init__": "⊢ p → p",
            ("intro h",): TacticOutcome(pp="h : p ⊢ p", done=False),
            ("intro h", "exact h"): TacticOutcome(pp="no goals", done=True),
            ("intro h", "bad"): TacticOutcome(pp=None, done=False, error="unknown tactic"),
        }
    }


def test_fake_backend_initial_state_pp():
    be = FakeLeanBackend(_tree())
    assert be.initial_pp("t") == "⊢ p → p"


def test_fake_backend_run_from_path_replays_and_applies():
    be = FakeLeanBackend(_tree())
    out = be.run("t", ("intro h",), "exact h")
    assert out.done and out.pp == "no goals" and out.error is None


def test_fake_backend_reports_error_for_invalid_tactic():
    be = FakeLeanBackend(_tree())
    out = be.run("t", ("intro h",), "bad")
    assert out.error == "unknown tactic" and out.pp is None
