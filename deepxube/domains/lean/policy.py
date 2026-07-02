from typing import Dict, List, Protocol, runtime_checkable


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
        missing: List[str] = [s for s in states if s not in self._cache]
        if missing:
            # de-duplicate while preserving order
            uniq = list(dict.fromkeys(missing))
            results = self._inner.top_k(uniq, k)
            for s, tactics in zip(uniq, results, strict=True):
                self._cache[s] = tactics
        return [self._cache[s] for s in states]
