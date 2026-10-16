#!/usr/bin/env bash
# One-command bootstrap: venv, dependencies, data, pipeline.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Project: $(pwd)"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.11+ and re-run." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "==> Creating virtual environment"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f data/raw/studentInfo.csv ]; then
    echo "==> No dataset found in data/raw - attempting download"
    python -m src.pipeline --download || {
        echo
        echo "Download failed. Place the seven OULAD CSVs in data/raw/ and re-run:"
        echo "    bash scripts/bootstrap.sh"
        exit 1
    }
else
    echo "==> Dataset already present in data/raw"
    python -m src.pipeline
fi

echo
echo "==> Running the test suite"
pytest

cat <<'MSG'

============================================================
Bootstrap complete.

Start the dashboard with:
    source .venv/bin/activate
    streamlit run app/Home.py
============================================================
MSG
