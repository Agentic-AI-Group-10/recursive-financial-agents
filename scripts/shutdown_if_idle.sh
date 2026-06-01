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

# 2. Check for active developer environments (VS Code remote, tmux, screen)
# VS Code Remote runs background node/server processes. Tmux/screen are used for persistent runs.
ACTIVE_VSCODE=$(pgrep -f "vscode-server" | wc -l)
ACTIVE_TMUX_SCREEN=$(pgrep -f "tmux|screen" | wc -l)

# 3. Check for active long-running Python processes
# In this workspace, quantitative trading backtests or agent loops run on Python.
# We exclude python grep/pgrep commands and this script itself.
ACTIVE_PYTHON_PROCS=$(pgrep -f "python3" | grep -v "shutdown_if_idle" | grep -v "grep" | wc -l)

# 4. Check CPU Load Average (5-minute average)
# If CPU is under heavy load (e.g. running high-frequency iterations or local model inference),
# the 5-minute load average will be elevated. Threshold is set to 0.15.
CPU_LOAD_5M=$(cat /proc/loadavg | awk '{print $2}')
CPU_THRESHOLD="0.15"
CPU_BUSY=$(awk -v load="$CPU_LOAD_5M" -v thresh="$CPU_THRESHOLD" 'BEGIN {print (load > thresh) ? "1" : "0"}')

log "Status: Active SSH: $ACTIVE_SSH_SESSIONS, VSCode: $ACTIVE_VSCODE, Tmux/Screen: $ACTIVE_TMUX_SCREEN, Python: $ACTIVE_PYTHON_PROCS, 5m-Load: $CPU_LOAD_5M"

# 5. Handle active users or processes with safety delays / cancel
if [ "$ACTIVE_SSH_SESSIONS" -gt 0 ]; then
    MSG="[Auto-Shutdown] Postponed: Active interactive SSH sessions ($ACTIVE_SSH_SESSIONS) detected. Will retry next schedule."
    log "$MSG"
    wall "$MSG"
    exit 0
fi

# NOTE: We do not automatically postpone on active VSCode remote node servers ($ACTIVE_VSCODE)
# or idle Tmux sessions ($ACTIVE_TMUX_SCREEN) if there are no live interactive SSH users
# and no active Python processes or CPU loads. This ensures that when a developer closes
# their laptop (disconnecting from SSH but leaving VS Code Remote servers running), 
# the VM can safely power off to avoid wasting money.

if [ "$ACTIVE_PYTHON_PROCS" -gt 0 ]; then
    MSG="[Auto-Shutdown] Postponed: Active Python processes ($ACTIVE_PYTHON_PROCS) are running (e.g. trading simulations)."
    log "$MSG"
    wall "$MSG"
    exit 0
fi

if [ "$CPU_BUSY" -eq 1 ]; then
    MSG="[Auto-Shutdown] Postponed: CPU is active (5m-load: $CPU_LOAD_5M > $CPU_THRESHOLD)."
    log "$MSG"
    wall "$MSG"
    exit 0
fi

# 6. Final warning broadcast (wall)
log "No active users, developer sessions, Python processes, or significant CPU load found. Initiating shutdown sequence..."
wall "[Auto-Shutdown] NO ACTIVE USERS OR PROCESSES DETECTED. The VM will shut down in 1 minute. Run 'sudo shutdown -c' to cancel."

# 7. Shut down with a 1-minute delay to allow manual cancellation if a user just logged in
/usr/sbin/shutdown -h +1 "Scheduled nightly auto-shutdown."
log "Shutdown command issued for +1 minute."

