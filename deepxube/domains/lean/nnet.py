from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from deepxube.base.nnet_input import StateGoalIn
from deepxube.factories.nnet_input_factory import register_nnet_input
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
