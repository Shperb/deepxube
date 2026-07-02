from deepxube.domains.lean.types import LeanState, LeanGoal
from deepxube.domains.lean.nnet import LeanHeurIn, ByteTokenizer
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _domain():
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    be = FakeLeanBackend({"u@c:F.lean:t": {"__init__": "⊢ p"}})
    return LeanDomain(be, FakeTacticGenerator({}), corpus, k=2, model_name="m", seed=0)


def test_byte_tokenizer_pads_and_masks():
    tok = ByteTokenizer(max_len=8)
    ids, mask = tok.encode_batch(["ab", "abcdefghij"])
    assert ids.shape == (2, 8) and mask.shape == (2, 8)
    assert mask[0].sum() == 2 and mask[1].sum() == 8  # second is truncated to max_len


def test_lean_heur_in_to_np_shapes():
    d = _domain()
    nin = LeanHeurIn(d, max_len=16)
    states = [LeanState("t", (), "⊢ p", done=False)]
    goals = [LeanGoal(None)]  # empty goal
    arrays = nin.to_np(states, goals)
    # [s_ids, s_mask, g_ids, g_mask, g_empty]
    assert len(arrays) == 5
    assert arrays[0].shape == (1, 16) and arrays[1].shape == (1, 16)
    assert arrays[2].shape == (1, 16) and arrays[3].shape == (1, 16)
    assert arrays[4].shape == (1, 1) and arrays[4][0, 0] == 1.0  # empty-goal flag set


def test_lean_heur_in_relabeled_goal_flag_zero():
    d = _domain()
    nin = LeanHeurIn(d, max_len=16)
    arrays = nin.to_np([LeanState("t", (), "⊢ p", done=False)], [LeanGoal("h : p ⊢ p")])
    assert arrays[4][0, 0] == 0.0
