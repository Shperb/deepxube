import numpy as np
import torch
from deepxube.domains.lean.nnet import LeanHeurIn, LeanHeurNNet, EmbeddingEncoder
from deepxube.domains.lean.domain import LeanDomain
from deepxube.domains.lean.backend import FakeLeanBackend
from deepxube.domains.lean.policy import FakeTacticGenerator
from deepxube.domains.lean.corpus import Corpus, CorpusEntry
from deepxube.domains.lean.dojo_backend import TheoremRef


def _nin(max_len=16):
    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    d = LeanDomain(FakeLeanBackend({"u@c:F.lean:t": {"__init__": "⊢ p"}}),
                   FakeTacticGenerator({}), corpus, k=2, model_name="m", seed=0)
    return LeanHeurIn(d, max_len=max_len)


def test_forward_outputs_scalar_per_input():
    nin = _nin()
    nnet = LeanHeurNNet(nin, out_dim=1, q_fix=False,
                        encoder=EmbeddingEncoder(vocab=260, hidden=8), hidden=8, mlp_hidden=16)
    nnet.eval()
    s_ids = np.zeros((3, 16), dtype=np.int64)
    s_ids[:, 0] = 5
    s_mask = np.zeros((3, 16), dtype=np.int64)
    s_mask[:, 0] = 1
    g_ids = np.zeros((3, 16), dtype=np.int64)
    g_mask = np.zeros((3, 16), dtype=np.int64)
    g_empty = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
    inputs = [torch.from_numpy(a) for a in [s_ids, s_mask, g_ids, g_mask, g_empty]]
    out = nnet(inputs)  # HeurNNet.forward returns a list in eval
    y = out[0]
    assert y.shape == (3, 1)
    assert torch.all(y >= 0)  # clamped non-negative in _forward


def test_encoder_is_frozen_flag_respected():
    nin = _nin()
    nnet = LeanHeurNNet(nin, out_dim=1, q_fix=False,
                        encoder=EmbeddingEncoder(vocab=260, hidden=8), hidden=8, mlp_hidden=16, freeze_encoder=True)
    assert all(not p.requires_grad for p in nnet.encoder.parameters())
    assert any(p.requires_grad for p in nnet.head.parameters())
