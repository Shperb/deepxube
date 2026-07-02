from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TacticGenerator(Protocol):
    def top_k(self, states: List[str], k: int) -> List[List[str]]:
        """ For each pretty-printed state, return up to k candidate tactic strings, best first. """
        ...


class FakeTacticGenerator:
    """ Deterministic generator for tests. """

    def __init__(self, table: Dict[str, List[str]]):
        self._table = table

    def top_k(self, states: List[str], k: int) -> List[List[str]]:
        return [list(self._table.get(s, []))[:k] for s in states]


class CachedTacticGenerator:
    """ Wraps a TacticGenerator with a per-state cache; batches only uncached states in one call. """

    def __init__(self, inner: TacticGenerator):
        self._inner = inner
        self._cache: Dict[str, List[str]] = {}

    def top_k(self, states: List[str], k: int) -> List[List[str]]:
        # (Re)query states that are uncached or whose cached list has fewer than k entries
        # (so resampling with a larger k grows the candidate set). If the inner generator is
        # exhausted (returns fewer than k), the shorter list is cached and returned as-is.
        need: List[str] = [s for s in states if len(self._cache.get(s, [])) < k]
        if need:
            uniq = list(dict.fromkeys(need))  # de-duplicate while preserving order
            results = self._inner.top_k(uniq, k)
            for s, tactics in zip(uniq, results, strict=True):
                if len(tactics) >= len(self._cache.get(s, [])):
                    self._cache[s] = tactics
        return [self._cache[s][:k] for s in states]


class ReProverGenerator:
    """ ReProver ByT5 tactic generator (deterministic beam search). Lazily loads transformers/torch. """

    def __init__(self, model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
                 device: str = "cpu", max_length: int = 1024):
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from deepxube.domains.lean._optional import require
        tf = require("transformers", extra="lean")
        self._tokenizer = tf.AutoTokenizer.from_pretrained(self._model_name)
        self._model = tf.AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def top_k(self, states: List[str], k: int) -> List[List[str]]:
        import torch
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        out: List[List[str]] = []
        for state in states:
            enc = self._tokenizer(state, return_tensors="pt", truncation=True,
                                  max_length=self._max_length).to(self._device)
            with torch.no_grad():
                gen_ids = self._model.generate(
                    enc.input_ids,
                    max_length=self._max_length,
                    num_beams=k,
                    num_return_sequences=k,
                    do_sample=False,
                    length_penalty=0.0,
                    early_stopping=False,
                )
            tactics = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            out.append(list(dict.fromkeys(t.strip() for t in tactics if t.strip())))
        return out
