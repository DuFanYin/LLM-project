#!/usr/bin/env bash
# Create .venv, install requirements + ipykernel, register a Jupyter kernel for this project.
# Usage: ./setup.sh
# Skip kernel registration (e.g. CI): SKIP_JUPYTER_KERNEL=1 ./setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Creating venv at .venv ..."
  python3 -m venv "$VENV"
fi

echo "Upgrading pip and installing dependencies ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "${ROOT}/requirements.txt" ipykernel

if [[ "${SKIP_JUPYTER_KERNEL:-}" == "1" ]]; then
  echo "Skipping Jupyter kernel (SKIP_JUPYTER_KERNEL=1)."
else
  echo "Registering Jupyter kernel 'llm-project' ..."
  "$PY" -m ipykernel install --user --name=llm-project --display-name="Python (.venv LLM-project)"
fi

echo ""
echo "Setup finished."
echo "  Activate:  source ${ROOT}/.venv/bin/activate"
echo "  In Jupyter / VS Code / Cursor: pick kernel 'Python (.venv LLM-project)' or interpreter ${ROOT}/.venv/bin/python"
