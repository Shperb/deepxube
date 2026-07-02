from deepxube.domains.lean.policy import FakeTacticGenerator, CachedTacticGenerator


def test_cached_generator_grows_candidates_when_k_increases():
    calls = {"n": 0}

    class Counting(FakeTacticGenerator):
        def top_k(self, states, k):
            calls["n"] += 1
            return super().top_k(states, k)

    gen = CachedTacticGenerator(Counting({"p": ["a", "b", "c", "d"]}))
    assert gen.top_k(["p"], k=2) == [["a", "b"]]          # cache 2
    assert gen.top_k(["p"], k=2) == [["a", "b"]]          # served from cache
    assert calls["n"] == 1
    assert gen.top_k(["p"], k=4) == [["a", "b", "c", "d"]]  # k grew -> re-query
    assert calls["n"] == 2


def test_fake_generator_returns_top_k_per_state():
    gen = FakeTacticGenerator({"⊢ p": ["intro h", "simp", "exact h"]})
    out = gen.top_k(["⊢ p"], k=2)
    assert out == [["intro h", "simp"]]


def test_cached_generator_calls_underlying_once_per_state():
    calls = {"n": 0}

    class Counting(FakeTacticGenerator):
        def top_k(self, states, k):
            calls["n"] += 1
            return super().top_k(states, k)

    gen = CachedTacticGenerator(Counting({"⊢ p": ["a", "b"]}))
    assert gen.top_k(["⊢ p"], k=2) == [["a", "b"]]
    assert gen.top_k(["⊢ p"], k=2) == [["a", "b"]]
    assert calls["n"] == 1  # second call served from cache


def test_cached_generator_batches_only_uncached_states():
    gen = CachedTacticGenerator(FakeTacticGenerator({"⊢ p": ["a"], "⊢ q": ["b"]}))
    assert gen.top_k(["⊢ p"], k=1) == [["a"]]
    assert gen.top_k(["⊢ p", "⊢ q"], k=1) == [["a"], ["b"]]  # order preserved, ⊢ p from cache
