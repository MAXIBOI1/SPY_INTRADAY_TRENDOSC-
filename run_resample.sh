#!/usr/bin/env bash
# Run resample_to_timeframes using the project venv. Creates venv if missing.
set -e
cd "$(dirname "$0")"

if [[ ! -d venv ]]; then
  echo "Creating venv..."
  python3 -m venv venv
  venv/bin/pip install --upgrade pip
  venv/bin/pip install -r config/requirements.txt
fi

venv/bin/python3 -m data.resample_to_timeframes "$@"
