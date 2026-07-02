# Lean domain setup (WSL)

The Lean domain requires a Linux environment. On Windows, use WSL.

1. Install WSL + Ubuntu, then inside WSL install the Python env: `pip install deepxube[lean]`.
2. Install the Lean toolchain (`elan`); LeanDojo drives it. See https://leandojo.readthedocs.io.
3. Trace the target repo once (mathlib for training, MiniF2F for eval). See `scripts/lean/setup_wsl.sh`.
4. Build a corpus manifest with `scripts/lean/build_corpus.py` (writes `lean_corpus.json`).
5. Run the smoke test: `DEEPXUBE_LEAN_TESTS=1 pytest deepxube/tests/lean -m lean -v`.
6. Determinism: set `PYTHONHASHSEED=0`; the domain seeds sampling; ReProver decoding is deterministic (`do_sample=False`). Pin the Lean toolchain commit and the ReProver checkpoint revision.

## Training / solving

- Train (heuristic via HER value-learning):
  `deepxube train --domain lean.8k --heur leanheur --heur_type V --pathfind graph_v --update update_v_rl_her --dir lean_run/`
- Solve:
  `deepxube solve --domain lean.8k --heur leanheur --heur_file lean_run/heur.pt --heur_type V --pathfind graph_v.1B_1.0W --file eval.pkl --results lean_results/`

(Exact flag names for `--update`/`--pathfind` may vary; confirm with `deepxube train --help`.)
