""" Build a pickle of {"states": [...], "goals": [...]} for the Lean domain, one instance per theorem
in a corpus manifest. The pickle is consumed by `deepxube train --t_file ...` (evaluation during
training) and by `deepxube solve --file ...`.

Each state is the initial tactic state of a theorem (opened via LeanDojo); each goal is the empty
goal LeanGoal(None) (i.e. "prove it"). Requires the [lean] extra + a traced repo.

Usage:
    python scripts/lean/build_test_pickle.py --manifest lean_corpus.json --out eval.pkl [--k 8]
"""
import argparse
import pickle
from typing import List

from deepxube.domains.lean.domain import build_lean_domain
from deepxube.domains.lean.types import LeanState, LeanGoal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="corpus manifest JSON (see build_corpus.py)")
    ap.add_argument("--out", required=True, help="output pickle path")
    ap.add_argument("--k", type=int, default=8, help="top-k (only affects the domain object, not the instances)")
    args = ap.parse_args()

    domain = build_lean_domain(k=args.k, manifest=args.manifest)

    states: List[LeanState] = []
    for ref in domain.corpus.theorem_refs():
        pp = domain.backend.initial_pp(ref.theorem_id)
        states.append(LeanState(theorem_id=ref.theorem_id, tactic_path=(), pp=pp, done=False))
    goals: List[LeanGoal] = [LeanGoal(None) for _ in states]

    with open(args.out, "wb") as fh:
        pickle.dump({"states": states, "goals": goals}, fh)
    print(f"wrote {args.out} with {len(states)} instances")


if __name__ == "__main__":
    main()
