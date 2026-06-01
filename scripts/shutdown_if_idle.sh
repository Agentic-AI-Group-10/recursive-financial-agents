#!/bin/bash
# ==============================================================================
# GCP VM Auto-Shutdown Script (with Safety & Idle Verification)
# Location: /home/ow9800/recursive-financial-agents/scripts/shutdown_if_idle.sh
# ==============================================================================
# This script is designed to run via cron to automatically power off the VM
# if no active user sessions or critical background processes are running.
# It logs actions to /var/log/auto-shutdown.log.

LOG_FILE="/var/log/auto-shutdown.log"

# Ensure the log file exists and is writable
touch "$LOG_FILE" 2>/dev/null || LOG_FILE="/tmp/auto-shutdown.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - $1" >> "$LOG_FILE"
}

log "Starting auto-shutdown check..."

# 1. Check for active SSH sessions
# We count lines in 'who' that contain 'pts/' indicating interactive terminals.
ACTIVE_SSH_SESSIONS=$(who | grep 'pts/' | wc -l)

# 2. Check for active long-running Python processes
# In this workspace, quantitative trading backtests or agent loops run on Python.
# We exclude python grep/pgrep commands and this script itself.
ACTIVE_PYTHON_PROCS=$(pgrep -f "python3" | grep -v "shutdown_if_idle" | grep -v "grep" | wc -l)

log "Status: Active SSH sessions: $ACTIVE_SSH_SESSIONS, Active Python processes: $ACTIVE_PYTHON_PROCS"

# 3. Handle active users or processes with safety delays / cancel
if [ "$ACTIVE_SSH_SESSIONS" -gt 0 ]; then
    MSG="[Auto-Shutdown] Postponed: Active SSH sessions ($ACTIVE_SSH_SESSIONS) detected. Will retry next schedule."
    log "$MSG"
    wall "$MSG"
    exit 0
fi

if [ "$ACTIVE_PYTHON_PROCS" -gt 0 ]; then
    MSG="[Auto-Shutdown] Postponed: Active Python processes ($ACTIVE_PYTHON_PROCS) are running (e.g. trading simulations)."
    log "$MSG"
    wall "$MSG"
    exit 0
fi

# 4. Final warning broadcast (wall)
log "No active users or processes found. Initiating shutdown sequence..."
wall "[Auto-Shutdown] NO ACTIVE USERS OR PROCESSES DETECTED. The VM will shut down in 1 minute. Run 'sudo shutdown -c' to cancel."

# 5. Shut down with a 1-minute delay to allow manual cancellation if a user just logged in
/usr/sbin/shutdown -h +1 "Scheduled nightly auto-shutdown."
log "Shutdown command issued for +1 minute."
