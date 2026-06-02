#!/usr/bin/env bash
# ==============================================================================
# Automated Recursive Strategy Evolution Multi-Trial Script
# Location: /home/ow9800/recursive-financial-agents/scripts/run_automated_trials.sh
# ==============================================================================
# This script automates sequential, multi-generation comparative trials across 
# different LLM Architect models and data regimes (Filtered vs. Unfiltered).
# It handles Google AI Studio models and DeepInfra open-weights models sequentially.
# All outputs are piped to dedicated logs in the `logs/` directory.
# ==============================================================================

# Exit on absolute terminal failures, but allow the loop to proceed to next trials
# if an individual trial encounters an error.
set -u

# Ensure workspace paths
CWD="/home/ow9800/recursive-financial-agents"
cd "$CWD" || exit 1

# Create logs directory if it doesn't exist
mkdir -p "$CWD/logs"

# Define configurations
GEMINI_FLASH_GENS=100 # 100 cycles for Gemini Flash trials (high-speed marathon)
GEMINI_PRO_GENS=100   # 100 cycles for Gemini Pro (now safe on Paid Tier)
DEEPINFRA_GENS=100    # 100 cycles for DeepInfra (now running overnight marathon)
PACING_DELAY_FLASH=4.0 # 4.0s delay for Gemini Flash
PACING_DELAY_PRO=4.0   # 4.0s delay for Gemini Pro (now unlocked on Paid Tier)

# Log files for session tracking
STATUS_LOG="$CWD/logs/trials_session_status.log"

echo "================================================================================" >> "$STATUS_LOG"
echo "Starting Automated Comparative Trials Session: $(date)" >> "$STATUS_LOG"
echo "================================================================================" >> "$STATUS_LOG"

# Helper function to run a trial
run_trial() {
    local provider="$1"
    local model="$2"
    local regime="$3" # "filtered" or "unfiltered"
    local gens="$4"
    local model_slug="${model//\//_}" # replace '/' with '_' for safe file naming
    local log_file="$CWD/logs/trial_${provider}_${model_slug}_${regime}.log"
    local backup_file="$CWD/database/backup_trial_${provider}_${model_slug}_${regime}.sql"
    local telemetry_backup="$CWD/telemetry_backup/telemetry_${provider}_${model_slug}_${regime}"
    
    # Check if this trial has already been completed and backed up
    if [ -f "$backup_file" ]; then
        echo "================================================================================"
        echo "⏭️  SKIPPING TRIAL: [Provider: $provider] [Model: $model] [Regime: $regime]"
        echo "Reason: Backup file already exists at $backup_file"
        echo "================================================================================"
        return 0
    fi

    # Select datasets based on regime
    local dataset_normal="data/simulation_pack_${regime}_normal.csv"
    local dataset_stress="data/simulation_pack_${regime}.csv"
    
    echo "--------------------------------------------------------------------------------"
    echo "🚀 RUNNING TRIAL: [Provider: $provider] [Model: $model] [Regime: $regime] [Generations: $gens]"
    echo "--------------------------------------------------------------------------------"
    echo "Logs are being streamed to: $log_file"
    
    echo "START: $provider | $model | $regime | $(date)" >> "$STATUS_LOG"
    
    # 1. Reset active logic workspace to baseline (main branch)
    echo "🔄 Resetting active logic file to the baseline (main branch)..."
    git checkout main -- "$CWD/strategies/strategy_logic.py"
    
    # 2. Clean up previous evolutionary git branches to avoid branch name conflicts
    echo "🌿 Cleaning up previous local evolutionary git branches..."
    git branch | grep "evolution_gen_" | xargs git branch -D 2>/dev/null || true
    
    # 3. Clear active telemetry folder to ensure we capture clean charts
    echo "🧹 Clearing active telemetry folder..."
    rm -rf "$CWD/telemetry/"*
    
    # 4. Run the orchestrator with python3
    if [ "$provider" == "google" ]; then
        local delay="$PACING_DELAY_FLASH"
        if [ "$model" == "gemini-2.5-pro" ] || [ "$model" == "gemini-1.5-pro" ]; then
            delay="$PACING_DELAY_PRO"
        fi
        
        python3 "$CWD/engine/orchestrator.py" \
            --generations "$gens" \
            --dataset-normal "$dataset_normal" \
            --dataset-stress "$dataset_stress" \
            --architect-model "$model" \
            --pacing-delay "$delay" \
            --skip-pr > "$log_file" 2>&1
    else
        python3 "$CWD/engine/orchestrator.py" \
            --generations "$gens" \
            --dataset-normal "$dataset_normal" \
            --dataset-stress "$dataset_stress" \
            --architect-model "$model" \
            --pacing-delay "1.0" \
            --skip-pr > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $provider | $model | $regime completed cleanly."
        echo "END: $provider | $model | $regime | SUCCESS | $(date)" >> "$STATUS_LOG"
    else
        echo "❌ FAILURE: $provider | $model | $regime exited with code $exit_code."
        echo "END: $provider | $model | $regime | FAILED (Code: $exit_code) | $(date)" >> "$STATUS_LOG"
    fi
    
    # 5. Backup the Postgres database results for this trial
    echo "📦 Backing up database results to: $backup_file"
    pg_dump -U ow9800 -d recursive_trading -F p -f "$backup_file"
    
    # 6. Archive the telemetry charts for this trial
    echo "📁 Archiving telemetry charts to $telemetry_backup..."
    mkdir -p "$telemetry_backup"
    cp -r "$CWD/telemetry/"* "$telemetry_backup/" 2>/dev/null || true
    
    # 7. Reset/truncate the database tables to clear them for the next trial run
    echo "🧹 Resetting database tables for the next trial run..."
    psql -U ow9800 -d recursive_trading -c "TRUNCATE runs, lessons_learned, strategies CASCADE;"
    
    # Brief physical sleep before kicking off next model to cool down API sockets
    sleep 5
}

# ==============================================================================
# SECTION 1: GOOGLE GEMINI COMPARATIVE TRIALS (100 Cycles Each)
# ==============================================================================

# --- Trial 1: Gemini 2.5 Pro (Flagship Reasoning) ---
# Caps Gemini Pro to 10 generations per trial (2 trials = 20 generations = 80 requests)
# to stay safely below the 50 RPD (Requests Per Day) quota on the Free Tier.
run_trial "google" "gemini-2.5-pro" "filtered" "$GEMINI_PRO_GENS"
run_trial "google" "gemini-2.5-pro" "unfiltered" "$GEMINI_PRO_GENS"

# --- Trial 2: Gemini 2.5 Flash ("Gemini Lite") ---
run_trial "google" "gemini-2.5-flash" "filtered" "$GEMINI_FLASH_GENS"
run_trial "google" "gemini-2.5-flash" "unfiltered" "$GEMINI_FLASH_GENS"


# ==============================================================================
# SECTION 2: DEEPINFRA OPEN-WEIGHTS TRIALS (15 Cycles Each - Budget Guard)
# ==============================================================================
# NOTE: Uses 15 cycles to give a meaningful representation of their 
# evolutionary capability while respecting API costs and driving-home time limits.
# Change DEEPINFRA_GENS at the top of the file to 100 for overnight runs.

# --- Llama Family (Meta) ---
# 1. Llama 3.3 70B (The highly robust flagship generalist)
run_trial "deepinfra" "meta-llama/Llama-3.3-70B-Instruct" "filtered" "$DEEPINFRA_GENS"
# 2. Llama 3.1 8B (Ultra-fast, light-weight, cheap)
run_trial "deepinfra" "meta-llama/Meta-Llama-3.1-8B-Instruct" "filtered" "$DEEPINFRA_GENS"

# --- DeepSeek Family ---
# 1. DeepSeek-V3 (671B MoE, coding & reasoning masterpiece)
run_trial "deepinfra" "deepseek-ai/DeepSeek-V3" "filtered" "$DEEPINFRA_GENS"
# 2. DeepSeek-Coder-V2 (The gold standard for code generation)
run_trial "deepinfra" "deepseek-ai/DeepSeek-Coder-V2-Instruct" "filtered" "$DEEPINFRA_GENS"

# --- Qwen Family (Alibaba) ---
# 1. Qwen 2.5 72B (Incredible performance across quantitative tasks)
run_trial "deepinfra" "Qwen/Qwen2.5-72B-Instruct" "filtered" "$DEEPINFRA_GENS"
# 2. Qwen 2.5 Coder 32B (The premier dedicated coding model)
run_trial "deepinfra" "Qwen/Qwen2.5-Coder-32B-Instruct" "filtered" "$DEEPINFRA_GENS"

# --- GLM Family (Zhipu AI) ---
# 1. GLM-4 9B (Ultra-fast, highly responsive)
run_trial "deepinfra" "THUDM/glm-4-9b-chat" "filtered" "$DEEPINFRA_GENS"

# --- Kimi Family (Moonshot AI) ---
# 1. Kimi K2.5 (Superb long-context reasoning)
run_trial "deepinfra" "moonshotai/Kimi-K2.5" "filtered" "$DEEPINFRA_GENS"


echo "================================================================================" >> "$STATUS_LOG"
echo "All automated trials completed: $(date)" >> "$STATUS_LOG"
echo "================================================================================" >> "$STATUS_LOG"
echo "🎉 ALL COMPARATIVE TRIALS FINISHED! Check logs in $CWD/logs/"
