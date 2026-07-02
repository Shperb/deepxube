import os
import json
import pytest

pytestmark = pytest.mark.lean
RUN = os.environ.get("DEEPXUBE_LEAN_TESTS") == "1"


@pytest.mark.skipif(not RUN, reason="requires traced Lean repo + ReProver; set DEEPXUBE_LEAN_TESTS=1")
def test_solve_trivial_theorem_end_to_end(tmp_path):
    from deepxube.domains.lean.domain import build_lean_domain

    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"theorems": [{
        "url": "https://github.com/yangky11/lean4-example",
        "commit": "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f",
        "file_path": "Lean4Example.lean",
        "theorem_name": "hello_world",
        "proof_len": 1,
    }]}))
    domain = build_lean_domain(k=8, manifest=str(manifest), seed=0)

    states, goals = domain.sample_problem_instances([1])
    acts = domain.get_state_actions(states)
    assert len(acts[0]) >= 1
    nxt, tcs = domain.next_state([states[0]], [acts[0][0]])
    assert tcs == [1.0]
    assert domain.is_solved(nxt, goals) == [nxt[0].done]
