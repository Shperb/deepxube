from dataclasses import dataclass
from typing import List, Optional
import random

from deepxube.domains.lean.dojo_backend import TheoremRef


@dataclass(frozen=True)
class CorpusEntry:
    ref: TheoremRef
    proof_len: Optional[int] = None


class Corpus:
    """ A set of theorems with optional human proof lengths, used for curriculum sampling. """

    def __init__(self, entries: List[CorpusEntry]):
        assert len(entries) > 0, "corpus must be non-empty"
        self._entries = entries

    def theorem_refs(self) -> List[TheoremRef]:
        return [e.ref for e in self._entries]

    def sample_theorems(self, num_steps_l: List[int], seed: int) -> List[CorpusEntry]:
        """ For each requested step count, deterministically pick a theorem near that proof length.

        :param num_steps_l: requested difficulties (one theorem returned per element)
        :param seed: base seed for reproducibility
        :return: list of CorpusEntry, one per requested step count
        """
        rng = random.Random(seed)
        with_len = [e for e in self._entries if e.proof_len is not None]
        picked: List[CorpusEntry] = []
        for i, steps in enumerate(num_steps_l):
            sub_rng = random.Random((seed, i, steps).__hash__())
            if with_len:
                # tolerance window widens if empty; deterministic tie-break by name
                window = sorted(with_len, key=lambda e: (abs((e.proof_len or 0) - steps), e.ref.theorem_name))
                near = [e for e in window if abs((e.proof_len or 0) - steps) <= max(1, steps // 4)] or window[:1]
                picked.append(near[sub_rng.randrange(len(near))])
            else:
                picked.append(self._entries[rng.randrange(len(self._entries))])
        return picked
