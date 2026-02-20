#!/bin/bash
# Unified CHPC experiment automation script.
# Usage: chpc_train.sh on|off|cron
#
# on   — Install a crontab entry that runs this script every 30 minutes.
# off  — Remove the crontab entry.
# cron — The actual work: pull results, regenerate plots, push code, queue jobs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
REMOTE="iferreira@lengau.chpc.ac.za"
REMOTE_DIR="/home/iferreira/lustre/distillation-decomposition"

mkdir -p "$LOG_DIR"

case "$1" in
  on)
    SCRIPT="$(realpath "$0")"
    # Remove any existing entry, then add fresh one
    (crontab -l 2>/dev/null | grep -v "chpc_train.sh"; echo "*/30 * * * * $SCRIPT cron >> $LOG_DIR/cron.log 2>&1") | crontab -
    echo "Cron installed (every 30 min). Logs: $LOG_DIR/cron.log"
    ;;
  off)
    crontab -l 2>/dev/null | grep -v "chpc_train.sh" | crontab -
    echo "Cron removed"
    ;;
  cron)
    echo "=== $(date) ==="

    # 1. Pull experiment results from CHPC
    echo "Pulling experiments from CHPC..."
    rsync -avz --delete \
      --include='experiments/***' --exclude='*' \
      "$REMOTE:$REMOTE_DIR/" "$PROJECT_DIR/"

    # 2. Regenerate plots locally (if plot_experiments.py exists)
    if [ -f "$PROJECT_DIR/plot_experiments.py" ]; then
      echo "Regenerating plots..."
      python "$PROJECT_DIR/plot_experiments.py"
    fi

    # 3. Push code to CHPC
    echo "Pushing code to CHPC..."
    rsync -avz --delete \
      --exclude='analysis' --exclude='.git' --exclude='.venv' \
      --exclude='experiments/' --exclude='data' --exclude='msc-cs' \
      "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

    # 4. Queue new/resumed jobs on CHPC
    echo "Queuing jobs on CHPC..."
    ssh "$REMOTE" "cd $REMOTE_DIR && python runner.py"

    echo "=== Done ==="
    ;;
  *)
    echo "Usage: $0 on|off|cron"
    exit 1
    ;;
esac
