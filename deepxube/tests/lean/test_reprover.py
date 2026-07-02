import os
import pytest


@pytest.mark.lean
@pytest.mark.skipif(os.environ.get("DEEPXUBE_LEAN_TESTS") != "1",
                    reason="downloads the ReProver model; set DEEPXUBE_LEAN_TESTS=1")
def test_reprover_generates_k_tactics():
    from deepxube.domains.lean.policy import ReProverGenerator
    gen = ReProverGenerator(model_name="kaiyuy/leandojo-lean4-tacgen-byt5-small", device="cpu")
    out = gen.top_k(["n : ℕ\n⊢ gcd n n = n"], k=4)
    assert len(out) == 1 and 1 <= len(out[0]) <= 4 and all(isinstance(t, str) for t in out[0])


def test_reprover_construction_is_lazy():
    # Constructing must not import transformers/torch or download the model.
    from deepxube.domains.lean.policy import ReProverGenerator
    gen = ReProverGenerator(model_name="kaiyuy/leandojo-lean4-tacgen-byt5-small", device="cpu")
    assert gen._model is None and gen._tokenizer is None
