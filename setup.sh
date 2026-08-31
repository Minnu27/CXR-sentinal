#!/usr/bin/env bash
# One-time local setup for VS Code / terminal use.
# Colab doesn't need this — it already has torch/torchvision preinstalled.
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done. In VS Code: Cmd/Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv/bin/python"
echo "Then run the 'Self-test' launch config (Run and Debug panel) to verify the pipeline."
