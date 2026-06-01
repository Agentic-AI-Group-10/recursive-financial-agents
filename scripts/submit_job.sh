#!/bin/bash
# ==============================================================================
# SLURM Job Submission Script for MIT Supercloud (The Foundry)
# Location: scripts/submit_job.sh
# ==============================================================================
# This script schedules the backtesting engine as a batch job on MIT Supercloud
# compute nodes, allowing automated execution of heavy LLM sentiment strategies.

#SBATCH -J spy_agent_backtest
#SBATCH -N 1                      # Request 1 node
#SBATCH --tasks-per-node=1        # Run 1 task per node
#SBATCH -t 01:00:00               # 1-hour wall time limit
#SBATCH -o logs/job_%j.out        # Standard output file (%j is the SLURM job ID)
#SBATCH -e logs/job_%j.err        # Standard error file
#SBATCH --exclusive               # Request exclusive access to nodes for telemetry accuracy

# 1. Setup workspace paths and environment
cd /home/ow9800/recursive-financial-agents || exit 1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create logs/ and telemetry/ directories if they don't exist
mkdir -p logs
mkdir -p telemetry

# 2. Load necessary Supercloud system modules
# On MIT Supercloud, Python & Anaconda environments are managed via modules.
echo "Loading system modules..."
module purge
module load anaconda/2023a-pytorch  # Load standard Python environment with scientific libs

# 3. Check arguments or use default simulation dataset
DATA_FILE=${1:-"data/simulation_pack_filtered_normal.csv"}
OUTPUT_FILE=${2:-"telemetry/results_filtered_normal.json"}
PLOT_FILE=${3:-"telemetry/equity_curve_normal.png"}
STRATEGY=${4:-"strategies.strategy_logic"}
HARDWARE_MODE=${5:-"cpu"} # "cpu" or "gpu"

echo "===================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Data File: $DATA_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Plot Path: $PLOT_FILE"
echo "Strategy: $STRATEGY"
echo "Hardware Profiling Mode: $HARDWARE_MODE"
echo "===================================================================="

# 4. Check if GPU requested and query card specifications
if [ "$HARDWARE_MODE" = "gpu" ]; then
    echo "Querying NVIDIA GPU State..."
    nvidia-smi
fi

# 5. Run the backtesting engine
python3 engine/backtest_engine.py \
    --data-file "$DATA_FILE" \
    --output "$OUTPUT_FILE" \
    --plot-path "$PLOT_FILE" \
    --strategy "$STRATEGY" \
    --hardware-mode "$HARDWARE_MODE" \
    --estimate-tokens

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===================================================================="
    echo "✅ Backtest Simulation Completed Successfully on SLURM Node!"
    echo "===================================================================="
else
    echo "===================================================================="
    echo "❌ Backtest Simulation Failed on SLURM Node with Exit Code: $EXIT_CODE"
    echo "===================================================================="
    exit $EXIT_CODE
fi
