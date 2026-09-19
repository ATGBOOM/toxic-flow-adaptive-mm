#!/usr/bin/env bash
# run_demo.sh — regenerates the mock fixture and runs the real pipeline
# end to end, unmodified, for a live interview demo.
#
# Usage: bash scripts/run_demo.sh
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== 1/6 generating mock fixtures ==="
python scripts/generate_mock_data.py

echo -e "\n=== 2/6 cleaning trades ==="
(cd src/data && python pipeline.py)

echo -e "\n=== 3/6 reconstructing order book ==="
python src/features/reconstructor.py

echo -e "\n=== 4/6 building feature matrix ==="
python src/features/build_features.py

echo -e "\n=== 5/6 training classifier + exporting predictions ==="
python src/evaluations/save_predictions.py

echo -e "\n=== 6/6 causal backtest + bootstrap CI ==="
python src/evaluations/bootstrap_eval.py

echo -e "\n=== done ==="
