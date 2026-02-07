#!/bin/bash
# Run informed-no-monitor experiment for trees concept

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Running experiment: 03_trees_informed_no_monitor"
echo "Project root: $PROJECT_ROOT"

# TODO: Implement experiment runner
# python "$PROJECT_ROOT/scripts/run_experiment.py" \
#     --config "$SCRIPT_DIR/config.yaml" \
#     --output-dir "$PROJECT_ROOT/data/outputs/03_trees_informed_no_monitor"

echo "Experiment runner not yet implemented"
exit 1
