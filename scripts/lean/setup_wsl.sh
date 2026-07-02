#!/usr/bin/env bash
set -euo pipefail
# Minimal one-time tracing helper. Pass the repo URL and commit to trace.
URL="${1:?repo url}"
COMMIT="${2:?commit hash}"
python - "$URL" "$COMMIT" <<'PY'
import sys
from lean_dojo import LeanGitRepo, trace
repo = LeanGitRepo(sys.argv[1], sys.argv[2])
trace(repo)  # one-time; downloads + traces
print("traced", repo)
PY
