from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from deepxube.domains.lean.backend import TacticOutcome
from deepxube.domains.lean._optional import require


@dataclass(frozen=True)
class TheoremRef:
    url: str
    commit: str
    file_path: str
    theorem_name: str

    @property
    def theorem_id(self) -> str:
        return f"{self.url}@{self.commit}:{self.file_path}:{self.theorem_name}"


class _Session:
    """ A live Dojo (via its context manager) plus TacticState objects reachable by tactic-path. """

    def __init__(self, cm: Any, dojo: Any, init_state: Any):
        self.cm = cm
        self.dojo = dojo
        self.states: Dict[Tuple[str, ...], Any] = {(): init_state}

    def close(self) -> None:
        try:
            self.cm.__exit__(None, None, None)
        except Exception:
            pass


class LeanDojoBackend:
    """ LeanBackend backed by LeanDojo, with a process-local LRU of open sessions.

    Sessions are not picklable, so they are dropped on pickling and rebuilt lazily.
    """

    def __init__(self, refs: Dict[str, TheoremRef], max_sessions: int = 8):
        self._refs = refs
        self._max_sessions = max_sessions
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        # idempotency cache so validating a tactic (get_state_actions) and realizing it
        # (next_state) costs a single run_tac. Process-local; rebuilt after pickling.
        self._outcome_cache: Dict[Tuple[str, Tuple[str, ...], str], TacticOutcome] = {}

    def __getstate__(self) -> Dict[str, Any]:
        return {"_refs": self._refs, "_max_sessions": self._max_sessions}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._refs = state["_refs"]
        self._max_sessions = state["_max_sessions"]
        self._sessions = OrderedDict()
        self._outcome_cache = {}

    @staticmethod
    def _is_lean_error(ld: Any, result: Any) -> bool:
        err_cls = getattr(ld, "LeanError", None)
        if err_cls is not None:
            return isinstance(result, err_cls)
        return type(result).__name__ == "LeanError"

    def _is_tactic_state(self, ld: Any, result: Any) -> bool:
        return (not isinstance(result, ld.ProofFinished)) and (not self._is_lean_error(ld, result))

    def _open(self, theorem_id: str) -> _Session:
        ld = require("lean_dojo", extra="lean")
        ref = self._refs[theorem_id]
        repo = ld.LeanGitRepo(ref.url, ref.commit)
        theorem = ld.Theorem(repo, ref.file_path, ref.theorem_name)
        cm = ld.Dojo(theorem)
        dojo, init_state = cm.__enter__()
        return _Session(cm, dojo, init_state)

    def _session(self, theorem_id: str) -> _Session:
        sess = self._sessions.get(theorem_id)
        if sess is not None:
            self._sessions.move_to_end(theorem_id)
            return sess
        sess = self._open(theorem_id)
        self._sessions[theorem_id] = sess
        while len(self._sessions) > self._max_sessions:
            _, evicted = self._sessions.popitem(last=False)
            evicted.close()
        return sess

    def _state_at(self, sess: _Session, tactic_path: Tuple[str, ...]) -> Any:
        ld = require("lean_dojo", extra="lean")
        path = tuple(tactic_path)
        prefix = path
        while prefix not in sess.states:
            prefix = prefix[:-1]
        cur = sess.states[prefix]
        for i in range(len(prefix), len(path)):
            result = sess.dojo.run_tac(cur, path[i])
            if self._is_tactic_state(ld, result):
                cur = result
                sess.states[path[: i + 1]] = cur
            else:
                raise RuntimeError(f"replay of {path[: i + 1]} did not yield a tactic state: {result}")
        return cur

    def initial_pp(self, theorem_id: str) -> str:
        sess = self._session(theorem_id)
        return str(sess.states[()].pp)

    def run(self, theorem_id: str, tactic_path: Tuple[str, ...], tactic: str) -> TacticOutcome:
        key: Tuple[str, Tuple[str, ...], str] = (theorem_id, tuple(tactic_path), tactic)
        cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached

        ld = require("lean_dojo", extra="lean")
        sess = self._session(theorem_id)
        cur = self._state_at(sess, tuple(tactic_path))
        result = sess.dojo.run_tac(cur, tactic)
        if isinstance(result, ld.ProofFinished):
            outcome = TacticOutcome(pp="no goals", done=True, error=None)
        elif self._is_lean_error(ld, result):
            outcome = TacticOutcome(pp=None, done=False, error=str(getattr(result, "error", result)))
        else:
            sess.states[tuple(tactic_path) + (tactic,)] = result
            outcome = TacticOutcome(pp=result.pp, done=False, error=None)

        self._outcome_cache[key] = outcome
        return outcome
