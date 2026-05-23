#!/bin/bash
# constraint-synth quickstart — render melody presets to WAV
set -e
echo "🔊 Constraint Synth — Quick Start"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pip install -e . --quiet 2>/dev/null || true

export PYTHONPATH="$SCRIPT_DIR"
python3 examples/demo_synth.py
echo "✅ constraint-synth works!"
