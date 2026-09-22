#!/usr/bin/env bash
#
# Deploy this repository to the Hugging Face Space `rasyidev/print-cost`.
#
# WHY THIS IS A SCRIPT AND NOT A CI-ONLY JOB
# ------------------------------------------
# The live Space runs PRE-REFACTOR code (plan F6): its src/helper.py is the old
# monolith, src/services/ does not exist there, and its Dockerfile entrypoint is
# `streamlit run main.py`, so the FastAPI app is never served. Pushing this
# repo's main to the Space syncs the code; it does NOT by itself make /health
# return JSON, because the Space's Dockerfile still starts Streamlit. Switching
# the served app is a deliberate follow-up (see "SERVED APP" below).
#
# Requires: git, git-lfs (optional), an HF token with WRITE access to the Space.
#   export HF_TOKEN=hf_...      # never commit this
#
# Usage:
#   bash scripts/deploy_space.sh              # dry run: shows what would change
#   bash scripts/deploy_space.sh --apply      # actually push to the Space
#   bash scripts/deploy_space.sh --apply --ci # non-interactive (for CI)
#
# SERVED APP (decide before flipping):
#   A. Keep Streamlit (current): the Space keeps working exactly as today, and
#      the deployed code becomes the refactored code in this repo. /health and
#      /openapi.json still return the Streamlit HTML page.
#   B. Serve FastAPI: change the Space's Dockerfile entrypoint to
#      `uvicorn main-fastapi:app --host 0.0.0.0 --port 8501`, keep app_port 8501
#      in README.md, and add `COPY main-fastapi.py /app`. /health then returns
#      {"status":"healthy","ml_model_version":"1.0.0","ml_model_loaded":true}
#      and /openapi.json exists — but the Streamlit upload form stops being the
#      served UI.
#   This script does neither automatically: option B changes what visitors see,
#   so it needs an explicit decision.

set -euo pipefail

SPACE_ID="${SPACE_ID:-rasyidev/print-cost}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
CI=0

for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --ci) CI=1 ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------
log "1/5 Checking the model artifact against the pin"
# ---------------------------------------------------------------------------
# A pointer file in models/ would deploy a Space that cannot predict anything.
python3 "${REPO_ROOT}/scripts/fetch_model.py" --check

# ---------------------------------------------------------------------------
log "2/5 Checking Hugging Face credentials"
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  cat >&2 <<'EOF'
HF_TOKEN is not set.

Nothing has been deployed. To deploy yourself:

  1. Create a fine-grained Hugging Face token with WRITE access to the Space:
       https://huggingface.co/settings/tokens
  2. export HF_TOKEN=hf_xxxxxxxx
  3. Re-run:  bash scripts/deploy_space.sh --apply

Or set the repository secret HF_TOKEN in GitHub (Settings -> Secrets and
variables -> Actions) and run the "CI" workflow manually: its deploy-space job
pushes main to the Space.
EOF
  exit 3
fi

if [[ "$APPLY" -eq 0 ]]; then
  log "DRY RUN — no push will happen (pass --apply to deploy)"
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# ---------------------------------------------------------------------------
log "3/5 Cloning the Space (${SPACE_ID})"
# ---------------------------------------------------------------------------
if ! git clone --quiet "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}" "${WORKDIR}/space"; then
  echo "FAIL: could not clone ${SPACE_URL}. Check the token's write scope." >&2
  exit 4
fi

# ---------------------------------------------------------------------------
log "4/5 Syncing this repository into the Space checkout"
# ---------------------------------------------------------------------------
# Everything tracked in git except the Space's own .git and the repo's CI config
# (GitHub Actions files are meaningless on the Space).
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude '.venv/' \
  --exclude 'htmlcov/' \
  --exclude '.pytest_cache/' \
  --exclude '.coverage' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  "${REPO_ROOT}/" "${WORKDIR}/space/"

cd "${WORKDIR}/space"
git add -A
if git diff --cached --quiet; then
  log "The Space already matches this repository. Nothing to deploy."
  exit 0
fi

echo
git diff --cached --stat
echo
git status --short

if [[ "$APPLY" -eq 0 ]]; then
  log "Dry run complete — the changes above were NOT pushed."
  exit 0
fi

# ---------------------------------------------------------------------------
log "5/5 Pushing to the Space"
# ---------------------------------------------------------------------------
git -c user.name="${GIT_AUTHOR_NAME:-rasyidev}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-habib.rasyid11@gmail.com}" \
    commit --quiet -m "Sync from github.com/rasyidev/print-cost@$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)"
git push --quiet origin HEAD:main

log "Pushed. The Space will rebuild automatically."
cat <<EOF

Verify once the build finishes (a cold rebuild takes a few minutes):

  curl -sS https://rasyidev-print-cost.hf.space/ | head -c 200

With the current Streamlit entrypoint you should see the Streamlit HTML page,
which means the upload form still works. If you switched the Dockerfile to
uvicorn (option B above), verify instead:

  curl -sS https://rasyidev-print-cost.hf.space/health
  curl -sS -o /dev/null -w '%{http_code}\n' https://rasyidev-print-cost.hf.space/openapi.json

The informational space-smoke job in .github/workflows/ci.yml reports which of
the two the live Space is serving.
EOF
