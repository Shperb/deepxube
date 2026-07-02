from typing import List, Tuple, Optional, Type, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn, Tensor

from deepxube.base.nnet_input import StateGoalIn
from deepxube.base.heuristic import HeurNNet
from deepxube.base.factory import DelimParser
from deepxube.factories.nnet_input_factory import register_nnet_input
from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.domains.lean.types import LeanState, LeanGoal

if TYPE_CHECKING:
    from deepxube.domains.lean.domain import LeanDomain


class ByteTokenizer:
    """ Deterministic byte-level tokenizer used for tests and as a fallback. Not the ReProver tokenizer. """

    def __init__(self, max_len: int = 512, pad_id: int = 0):
        self.max_len = max_len
        self.pad_id = pad_id

    def encode_batch(self, texts: List[str]) -> Tuple[NDArray, NDArray]:
        ids = np.full((len(texts), self.max_len), self.pad_id, dtype=np.int64)   # [B, L]
        mask = np.zeros((len(texts), self.max_len), dtype=np.int64)              # [B, L]
        for i, text in enumerate(texts):
            b = text.encode("utf-8")[: self.max_len]
            for j, byte in enumerate(b):
                ids[i, j] = byte + 1  # reserve 0 for pad
                mask[i, j] = 1
        return ids, mask


@register_nnet_input("lean", "lean_text_sg")
class LeanHeurIn(StateGoalIn["LeanDomain", LeanState, LeanGoal]):
    """ Tokenizes state pp and goal pp separately for the frozen-encoder heuristic. """

    def __init__(self, domain: "LeanDomain", max_len: int = 512):
        super().__init__(domain)
        self.max_len = max_len
        self._tok = ByteTokenizer(max_len=max_len)

    def get_input_info(self) -> int:
        return self.max_len

    def to_np(self, states: List[LeanState], goals: List[LeanGoal]) -> List[NDArray]:
        s_ids, s_mask = self._tok.encode_batch([s.pp for s in states])
        g_texts = [("" if g.is_empty() else g.target_pp) for g in goals]
        g_ids, g_mask = self._tok.encode_batch([t or "" for t in g_texts])
        g_empty = np.array([[1.0 if g.is_empty() else 0.0] for g in goals], dtype=np.float32)  # [B, 1]
        return [s_ids, s_mask, g_ids, g_mask, g_empty]


def _mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    # hidden: [B, L, H], mask: [B, L] -> [B, H]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


class EmbeddingEncoder(nn.Module):
    """ Tiny encoder for tests: embedding producing per-token 'hidden states'. Returns [B, L, H]. """

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.hidden_size = hidden

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.embed(input_ids.long())  # [B, L, H]


class ByT5Encoder(nn.Module):
    """ Real encoder: the ByT5 encoder from the ReProver model, frozen by default. Loads transformers lazily. """

    def __init__(self, model_name: str):
        super().__init__()
        from deepxube.domains.lean._optional import require
        tf = require("transformers", extra="lean")
        full = tf.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.encoder = full.get_encoder()
        self.hidden_size = full.config.d_model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        return out.last_hidden_state  # [B, L, H]


@heuristic_factory.register_class("leanheur")
class LeanHeurNNet(HeurNNet[LeanHeurIn]):
    """ h(S, G) = MLP([enc(S); enc(G_or_empty)]) -> cost-to-go. Encoder shared and frozen by default. """

    @staticmethod
    def nnet_input_type() -> Type[LeanHeurIn]:
        return LeanHeurIn

    def __init__(self, nnet_input: LeanHeurIn, out_dim: int, q_fix: bool,
                 encoder: Optional[nn.Module] = None, hidden: Optional[int] = None,
                 mlp_hidden: int = 512, freeze_encoder: bool = True):
        super().__init__(nnet_input, out_dim, q_fix)
        assert out_dim == 1 and not q_fix, "Lean heuristic is a scalar V-function"
        if encoder is None:
            encoder = ByT5Encoder(nnet_input.domain.model_name)
        self.encoder: nn.Module = encoder
        h: int = hidden if hidden is not None else int(getattr(encoder, "hidden_size"))
        self.hidden_size: int = h
        self.freeze_encoder: bool = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.empty_goal_vec = nn.Parameter(torch.zeros(h))  # learned representation of the empty goal
        self.head = nn.Sequential(
            nn.Linear(2 * h, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, out_dim),
        )

    def _encode(self, ids: Tensor, mask: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                hidden = self.encoder(ids, mask)  # [B, L, H]
        else:
            hidden = self.encoder(ids, mask)
        return _mean_pool(hidden, mask)  # [B, H]

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        s_ids, s_mask, g_ids, g_mask, g_empty = inputs  # [B,L]*4, [B,1]
        enc_s = self._encode(s_ids, s_mask)                      # [B, H]
        enc_g_real = self._encode(g_ids, g_mask)                 # [B, H]
        empty = g_empty.to(enc_s.dtype)                          # [B, 1]
        enc_g = empty * self.empty_goal_vec.unsqueeze(0) + (1.0 - empty) * enc_g_real  # [B, H]
        x = self.head(torch.cat([enc_s, enc_g], dim=1))          # [B, out_dim]
        return torch.clamp(x, min=0.0)


@heuristic_factory.register_parser("leanheur")
class LeanHeurParser(DelimParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("mlp", "mlp_hidden", int, "hidden width of the MLP head")
        self.add_argument("ft", "freeze_encoder", None, "freeze the encoder (flag)")

    @property
    def delim(self) -> str:
        return "_"
