#!/bin/bash
# Run workflow in background with nohup
#
# Usage:
#   ./run_workflow.sh                              # Run with default config
#   ./run_workflow.sh -c workflow_config_linux.yaml  # Use specific config
#   ./run_workflow.sh -s load_data build_users     # Run specific stages

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

LOG_FILE="workflow_output_$(date +%Y%m%d_%H%M%S).log"

nohup python -m agentic_recommender.workflow.workflow_runner "$@" > "$LOG_FILE" 2>&1 &

echo "Workflow started with PID $!"
echo "Log file: $LOG_FILE"
echo ""
echo "To follow the log:"
echo "  tail -f $LOG_FILE"
