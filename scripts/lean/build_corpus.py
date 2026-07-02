""" Emit a lean_corpus.json manifest of theorems for the Lean domain.

Each theorem entry: {"url", "commit", "file_path", "theorem_name", "proof_len"?}.
Replace the example entries with theorems extracted from your traced repo (e.g. the LeanDojo benchmark).
Uses only the standard library so it runs anywhere (no Lean required).
"""
import json
import sys


def main(out_path: str) -> None:
    theorems = [
        {"url": "https://github.com/yangky11/lean4-example",
         "commit": "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f",
         "file_path": "Lean4Example.lean", "theorem_name": "hello_world", "proof_len": 1},
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"theorems": theorems}, fh, indent=2)
    print(f"wrote {out_path} with {len(theorems)} theorems")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lean_corpus.json")
