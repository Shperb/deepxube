""" Generate MiniF2F train (Valid) / eval (Test) corpus manifests for the Lean domain.

Clones the LeanDojo-compatible miniF2F-lean4 repo at a pinned commit and extracts one theorem
per file (name = the `theorem`/`lemma` identifier). Writes <out_dir>/minif2f_valid.json and
<out_dir>/minif2f_test.json in the manifest schema build_lean_domain expects. Stdlib + git only.

Usage:
    python scripts/lean/build_minif2f_manifests.py --out-dir ~
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List

REPO = "https://github.com/yangky11/miniF2F-lean4"
COMMIT = "5746b7d6c47855ce1294bed87329618ff7f1bc31"
NAME_RE = re.compile(r"^(?:theorem|lemma)\s+([A-Za-z0-9_']+)", re.M)


def _theorems(repo_dir: str, split: str) -> List[Dict[str, str]]:
    files = subprocess.check_output(
        ["git", "-C", repo_dir, "ls-files", f"MiniF2F/{split}/*.lean"], text=True
    ).split()
    out: List[Dict[str, str]] = []
    for rel in files:
        with open(os.path.join(repo_dir, rel), encoding="utf-8") as fh:
            match = NAME_RE.search(fh.read())
        assert match is not None, f"no theorem/lemma declaration found in {rel}"
        out.append({"url": REPO, "commit": COMMIT, "file_path": rel, "theorem_name": match.group(1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".", help="directory to write the two manifest JSON files")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = os.path.join(tmp, "miniF2F-lean4")
        subprocess.check_call(["git", "clone", "--quiet", REPO, repo_dir])
        subprocess.check_call(["git", "-C", repo_dir, "checkout", "--quiet", COMMIT])

        names = {"Valid": set(), "Test": set()}
        for split, out_name in [("Valid", "minif2f_valid.json"), ("Test", "minif2f_test.json")]:
            thms = _theorems(repo_dir, split)
            names[split] = {t["theorem_name"] for t in thms}
            out_path = os.path.join(os.path.expanduser(args.out_dir), out_name)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"theorems": thms}, fh, indent=1)
            print(f"{split}: {len(thms)} theorems -> {out_path}")

        overlap = names["Valid"] & names["Test"]
        assert not overlap, f"valid/test overlap: {sorted(overlap)[:5]}"
        print(f"valid/test disjoint OK (overlap={len(overlap)})")


if __name__ == "__main__":
    main()
