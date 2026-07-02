import os
import pickle
import pytest


def test_backend_is_picklable_without_lean_dojo():
    # __getstate__/__setstate__ must round-trip refs and reset the session cache,
    # without requiring lean_dojo to be installed.
    from deepxube.domains.lean.dojo_backend import LeanDojoBackend, TheoremRef
    ref = TheoremRef("https://example/repo", "abc123", "F.lean", "thm")
    be = LeanDojoBackend({ref.theorem_id: ref}, max_sessions=3)
    be2 = pickle.loads(pickle.dumps(be))
    assert ref.theorem_id in be2._refs
    assert be2._max_sessions == 3
    assert len(be2._sessions) == 0


def test_theorem_id_is_stable_and_unique():
    from deepxube.domains.lean.dojo_backend import TheoremRef
    a = TheoremRef("u", "c", "F.lean", "t")
    b = TheoremRef("u", "c", "F.lean", "t")
    c = TheoremRef("u", "c", "F.lean", "other")
    assert a.theorem_id == b.theorem_id and a.theorem_id != c.theorem_id


@pytest.mark.lean
@pytest.mark.skipif(os.environ.get("DEEPXUBE_LEAN_TESTS") != "1",
                    reason="set DEEPXUBE_LEAN_TESTS=1 and provide a traced repo")
def test_dojo_backend_round_trip():
    from deepxube.domains.lean.dojo_backend import LeanDojoBackend, TheoremRef
    ref = TheoremRef(
        url="https://github.com/yangky11/lean4-example",
        commit="7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f",
        file_path="Lean4Example.lean",
        theorem_name="hello_world",
    )
    be = LeanDojoBackend({ref.theorem_id: ref}, max_sessions=2)
    assert isinstance(be.initial_pp(ref.theorem_id), str)
    out = be.run(ref.theorem_id, (), "rfl")
    assert (out.error is not None) or (out.pp is not None)
