# Lean domain setup (WSL)

The Lean domain requires a Linux environment. On Windows, use WSL.

1. Install WSL + Ubuntu, then inside WSL install the Python env: `pip install deepxube[lean]`.
2. Install the Lean toolchain (`elan`); LeanDojo drives it. See https://leandojo.readthedocs.io.
3. Trace the target repo once (mathlib for training, MiniF2F for eval). See `scripts/lean/setup_wsl.sh`.
4. Build a corpus manifest with `scripts/lean/build_corpus.py` (writes `lean_corpus.json`).
5. Run the smoke test: `DEEPXUBE_LEAN_TESTS=1 pytest deepxube/tests/lean -m lean -v`.
6. Determinism: set `PYTHONHASHSEED=0`; the domain seeds sampling; ReProver decoding is deterministic (`do_sample=False`). Pin the Lean toolchain commit and the ReProver checkpoint revision.

## Training / solving

The updater is selected automatically from `--heur_type` + `--pathfind` + the `--her` flag
(`get_updater`); there is no `--update` argument. `--heur_type V --pathfind graph_v --her`
resolves to the `update_v_rl_her` updater (goal-conditioned V-learning with HER).

- Train (heuristic via HER value-learning):
  `deepxube train --domain lean.8k --heur leanheur --heur_type V --pathfind graph_v --her --step_max 30 --dir lean_run/ [--procs P --search_itrs S --up_itrs U --batch_size B --max_itrs M]`
- Evaluate during training: add `--t_file <instances.pkl> --t_pathfinds graph_v.1B_1.0W --t_init`,
  where the pickle holds `{"states": [...], "goals": [...]}` (build it with `scripts/lean/build_test_pickle.py`).
- Solve a saved instance set:
  `deepxube solve --domain lean.8k --heur leanheur --heur_file lean_run/heur.pt --heur_type V --pathfind graph_v.1B_1.0W --file eval.pkl --results lean_results/`

Note: `graph_v` in training must effectively generate one instance per iteration (the updater assumes this).
