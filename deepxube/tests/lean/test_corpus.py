from deepxube.domains.lean.dojo_backend import TheoremRef
from deepxube.domains.lean.corpus import Corpus, CorpusEntry


def _corpus():
    entries = [
        CorpusEntry(TheoremRef("u", "c", "F.lean", f"t{n}"), proof_len=n)
        for n in [1, 2, 3, 10, 11, 12]
    ]
    return Corpus(entries)


def test_sample_is_deterministic_given_seed():
    c = _corpus()
    a = c.sample_theorems([2, 2, 2], seed=0)
    b = c.sample_theorems([2, 2, 2], seed=0)
    assert [e.ref.theorem_name for e in a] == [e.ref.theorem_name for e in b]


def test_sample_prefers_matching_proof_length():
    c = _corpus()
    picked = c.sample_theorems([11], seed=0)
    assert picked[0].proof_len in {10, 11, 12}


def test_sample_returns_requested_count():
    c = _corpus()
    assert len(c.sample_theorems([1, 5, 12], seed=3)) == 3
