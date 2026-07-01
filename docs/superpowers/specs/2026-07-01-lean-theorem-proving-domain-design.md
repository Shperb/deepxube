# Lean Theorem-Proving Domain — Design

Date: 2026-07-01
Status: Approved (design), pending implementation plan

## 1. Objective

Add a theorem-proving domain to DeepXube in which:

- states are Lean 4 proof (tactic) states,
- the action set at each state is the top-*k* tactics produced by a pretrained state-of-the-art tactic-generation policy (ReProver), where *k* is a configuration parameter,
- a heuristic function is *learned* with DeepXube's reinforcement-learning loop, using Hindsight Experience Replay (HER) to produce dense training signal from mostly-failing proof searches,
- problems are attempted with the existing batch/weighted graph search.

Scope for this iteration: environment (domain) + solving + heuristic training. Training on Lean 4 mathlib; evaluation on MiniF2F.

The design maps onto existing DeepXube abstractions with minimal new machinery: the pretrained policy becomes the domain's action model (`ActsEnum.get_state_actions`), and HER is provided end-to-end by the existing `update_v_rl_her` updater, which only requires the domain to be `GoalSampleableFromState`.

## 2. Key decisions (settled during brainstorming)

| Decision | Choice |
| --- | --- |
| Scope | Environment + solving + training |
| Lean interaction + policy | LeanDojo + ReProver (Lean 4), run under WSL |
| Corpus | LeanDojo mathlib benchmark (train) / MiniF2F (eval) |
| Session model | Replay-based, process-local `Dojo` pool (picklable, `procs>1` safe) |
| Goal space / HER | Goal-conditioned tactic state: `h(S, G)`; deploy with `G = empty` |
| Heuristic encoder | Frozen, shared ReProver encoder + trainable MLP head |
| Transition cost | Unit cost `1.0` per tactic (cost-to-go = number of tactics) |
| Heuristic type / search | V-heuristic (`q_fix=False`) with `graph_v` search |

## 3. Module layout

New subpackage `deepxube/domains/lean/`. Registration decorators execute on import (consistent with the recursive domain import described in the README). **All heavy imports (`lean_dojo`, `torch`, `transformers`) are lazy inside methods** so that `domain_info` and other non-Lean commands stay fast and do not require the optional dependencies.

- `domain.py` — `LeanState`, `LeanAction`, `LeanGoal`, `LeanDomain`, `LeanParser`, registration.
- `session.py` — `DojoPool`: process-local LRU pool of LeanDojo `Dojo`s keyed by theorem, tactic-path replay, and a top-*k* tactic cache keyed by state text.
- `policy.py` — `ReProverPolicy`: loads the HF ReProver model/tokenizer, batched top-*k* tactic generation for a list of tactic states.
- `nnet.py` — `LeanHeurIn` (`NNetInput`) + `LeanHeurNNet` (heuristic), registered via `register_nnet_input` and `heuristic_factory`.
- `corpus.py` — enumerate theorems from traced mathlib (train) / MiniF2F (eval), with human proof lengths for curriculum bucketing.

## 4. State / Action / Goal representations

### `LeanState`
Fields: `theorem_id: str`, `tactic_path: tuple[str, ...]`, `pp: str`, `done: bool`.

- Fully serializable, hence picklable for `procs>1` and mid-search checkpointing.
- `__hash__ = hash((theorem_id, pp))`; `__eq__` on `(theorem_id, pp)`. Content-based identity is exactly what the CLOSED set and `Node.edge_dict` dedup require.
- `pp` is LeanDojo's pretty-printed tactic state.
- A shared `DEAD` sentinel state represents invalid-tactic outcomes.

### `LeanAction`
Field: `tactic: str`. `__hash__`, `__eq__`, `__repr__` on the string (repr prints the tactic so `viz --soln` shows the proof).

### `LeanGoal`
Field: `target_pp: Optional[str]`.

- `None` ⇒ empty tactic state ⇒ real proof objective (deployment goal).
- non-`None` ⇒ HER-relabeled reached tactic state.

### Transition cost
`1.0` per applied tactic ⇒ cost-to-go equals the number of tactics, matching DeepXube's cost-to-go semantics.

## 5. `LeanDomain`

Mixins: `ActsEnum`, `GoalSampleableFromState`, `StringToAct`.

- `sample_problem_instances(num_steps_l, times=None)` — sample theorems from the training corpus, bucketed by human proof length so `num_steps_l` drives a difficulty curriculum (fallback: uniform). Open each theorem's `Dojo` via the pool and return `(init_state, LeanGoal(None))` per theorem. Seeded sampling.
- `get_state_actions(states)` — `ReProverPolicy` top-*k* tactics per state, served through the top-*k* cache. `DEAD`/`done` states return `[]` (dead/terminal leaves; `ActsEnum.expand` tolerates zero-children states).
- `next_state(states, actions)` — group inputs by `theorem_id`, get-or-open each `Dojo` in the process-local pool, replay `tactic_path`, run the tactic. Return next `LeanState` (`done=True` on `ProofFinished`) at cost `1.0`; invalid/error tactic → `DEAD` at cost `1.0`.
- `is_solved(states, goals)` — per pair: `G.target_pp is None` ⇒ `state.done`; else `state.pp == G.target_pp`.
- `sample_goal_from_state(states_start, states_goal)` — `LeanGoal(target_pp=s.pp)` for each reached state. This single hook powers HER through `update_v_rl_her`.
- `string_to_action` / `string_to_action_help` — parse a typed tactic string into a `LeanAction` for `viz`.

Note: ReProver conditions on `S` only (goal-agnostic); only the heuristic is goal-conditioned. This is consistent with the pretrained model's interface.

## 6. Pretrained policy (ReProver) — the action model

`ReProverPolicy` wraps the HF ReProver checkpoint and returns, for a batch of tactic states, the top-*k* tactic strings (beam/sampling width = *k*) per state. Because the policy is the domain's action model (inside `get_state_actions`), search uses the plain `graph_v` (`PathFindActsEnum`) algorithm — no policy-function plumbing and no policy training in this iteration.

Caching by state `pp` is required: `get_state_actions` is hit at every node expansion and again during value-iteration re-expansion in the updater.

## 7. Learned heuristic `h(S, G)`

Frozen, shared ReProver encoder + trainable MLP head.

- `LeanHeurIn.to_np(states, goals)` tokenizes `S.pp` and `G.pp` **separately** with ReProver's tokenizer.
- `LeanHeurNNet` (`q_fix=False`, `out_dim=1`):
  - `enc(S) → [B, H]` and `enc(G) → [B, H]` via the frozen shared ReProver encoder, mean-pooled over non-pad tokens. `[B, H]` = batch × encoder hidden size.
  - empty goal (`target_pp=None`) uses a dedicated learned `[H]` parameter vector.
  - head: `MLP([enc(S); enc(G)]) → [B, 1]` cost-to-go, ReLU-clamped ≥ 0 (consistent with `HeurNNetParV._get_output`).
  - `enc(S)` is cached/reused from the policy's forward pass; only the MLP and the empty-goal vector train.
- Config flag to unfreeze the encoder for fine-tuning in a later iteration.

## 8. Training & HER wiring

- Updater: `update_v_rl_her` (V-learning + HER). Training/solving pathfinding: `graph_v`. Both already exist; no new updater or training-loop code.
- Per-iteration data flow: `sample_problem_instances` → run `graph_v` search with the current `h` → `_get_her_goals` keeps `G=∅` for solved instances and relabels the rest to the deepest reached tactic states (`sample_goal_from_state`) → replay-buffer value-iteration targets → train the MLP head. Failing searches still yield dense signal.
- Evaluation: periodic `test()` on MiniF2F, logging %solved / path cost / search iterations to TensorBoard.

## 9. Sessions, environment, determinism, testing

### Sessions
`DojoPool` maintains a bounded LRU of live `Dojo`s, process-local and rebuilt after pickling. Replay reconstructs any state from its `tactic_path`. An optional layer-C optimization (caching live `TacticState` objects by content-hash so replay only happens on a cache miss) is a drop-in later addition; the state representation does not change.

### Environment (WSL)
LeanDojo and the Lean 4 toolchain run under WSL on the Windows host. Tracing mathlib and MiniF2F is a one-time setup step. Provide a setup script and documentation. New optional dependencies (`lean-dojo`, `transformers`) are isolated to this subpackage and imported lazily.

### Determinism
Seed `random`, `numpy`, and `torch`; use deterministic ReProver decoding; seed theorem sampling. Document LeanDojo/subprocess nondeterminism and pin the Lean toolchain and ReProver checkpoint versions.

### Testing
- Unit tests with Lean mocked: `__hash__`/`__eq__`, `is_solved` for both goal modes, replay correctness, `DEAD` handling, HER goal sampling, curriculum bucketing.
- Lean-gated integration tests (skipped when Lean/WSL is unavailable): one theorem end-to-end round-trip (open → top-*k* → apply → solved detection); a tiny training smoke test.

## 10. Out of scope (this iteration)

- Training or fine-tuning the tactic policy (ReProver is used as a fixed pretrained action model).
- Unfreezing the heuristic encoder (flagged but off by default).
- Retrieval augmentation beyond what the chosen ReProver checkpoint provides.
- Layer-C live-`TacticState` caching (design-compatible future optimization).
