#!/usr/bin/env python3
# ==============================================================================
# Core Backtesting Engine
# Location: /home/ow9800/recursive-financial-agents/engine/backtest_engine.py
# ==============================================================================
# This engine executes historical daily backtests on the simulation packs,
# tracks strategy P&L, integrates hardware/token telemetry, and calculates
# the final Agentic ROI (AROI) Efficiency Score.

import os
import sys

# Ensure workspace root is in python path for dynamic importing
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import importlib
import argparse
import pandas as pd
import numpy as np

# Default Telemetry Pricing Constants
PRICE_PER_M_INPUT_TOKENS = 0.15   # USD per 1M input tokens (e.g. Gemini 1.5 Flash / Llama-3-8B)
PRICE_PER_M_OUTPUT_TOKENS = 0.60  # USD per 1M output tokens
CARBON_COST_PER_KJ = 0.0001        # Carbon offset/energy cost in USD per Kilojoule (KJ)
DEFAULT_CPU_TDP = 65.0             # Watts (TDP fallback for Intel/AMD CPUs)
DEFAULT_GPU_TDP = 250.0            # Watts (TDP fallback for NVIDIA GPUs)

def get_cpu_energy_joules():
    """Attempts to read Intel RAPL energy sensor. Returns None on failure."""
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    if os.path.exists(rapl_path):
        try:
            with open(rapl_path, "r") as f:
                uj = int(f.read().strip())
                return uj / 1000000.0  # Convert microjoules to Joules
        except (PermissionError, ValueError, IOError):
            pass
    return None

def get_gpu_power_watts():
    """Attempts to query NVIDIA GPU power draw via nvidia-smi. Returns None on failure."""
    import subprocess
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return float(res.decode().strip())
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None

class BacktestEngine:
    def __init__(self, data_path, initial_cash=10000.0, commission=0.0005, strategy_module="strategies.strategy_logic"):
        self.data_path = data_path
        self.initial_cash = initial_cash
        self.commission = commission
        self.strategy_module_name = strategy_module
        
        # Portfolio State
        self.cash = initial_cash
        self.shares = 0.0
        self.portfolio_value = initial_cash
        
        # Execution Telemetry
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.llm_calls = 0
        
        # Load and validate the data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Simulation pack file not found at: {data_path}")
            
        self.df = pd.read_csv(data_path, index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        
        # Verify columns
        required_cols = {'Close', 'news_context'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Simulation pack must contain columns: {required_cols}. Found: {self.df.columns.tolist()}")

    def load_strategy(self):
        """Loads or reloads the evolved strategy logic dynamically."""
        print(f"Loading strategy module: {self.strategy_module_name}...")
        try:
            # Force reload to get the latest evolved version of the strategy
            if self.strategy_module_name in sys.modules:
                importlib.reload(sys.modules[self.strategy_module_name])
                module = sys.modules[self.strategy_module_name]
            else:
                module = importlib.import_module(self.strategy_module_name)
                
            if not hasattr(module, "decide"):
                raise AttributeError("Strategy module is missing required 'decide()' function.")
            return module.decide
        except Exception as e:
            print(f"CRITICAL: Failed to load strategy module. Error: {e}")
            raise e

    def run(self, estimate_llm_tokens=False, model_tdp_mode="cpu"):
        """Executes the daily simulation loop."""
        decide_func = self.load_strategy()
        
        # Telemetry Baselines
        start_time = time.perf_counter()
        start_cpu_joules = get_cpu_energy_joules()
        
        trade_log = []
        portfolio_history = []
        prices = self.df['Close'].values
        news = self.df['news_context'].values
        timestamps = self.df.index.to_pydatetime()
        
        print(f"\n--- Launching Backtest Simulation ({len(self.df)} trading days) ---")
        
        for t in range(len(self.df)):
            current_price = float(prices[t])
            current_news = str(news[t])
            current_time = timestamps[t]
            
            # Look-ahead guard: Expose price history ONLY up to yesterday (index 0 to t-1)
            # If t=0, price history is empty
            price_history = list(prices[:t])
            
            # Step 1: Call Strategy
            decision = "HOLD"
            tokens_in = 0
            tokens_out = 0
            
            # Run safety wrapper around LLM / Strategy execution
            try:
                # Evolved strategies can return either a string action ("BUY", "SELL", "HOLD")
                # OR a dict/tuple with metadata e.g. {"decision": "BUY", "tokens_in": 1200, "tokens_out": 150}
                res = decide_func(current_price, price_history, current_news)
                
                if isinstance(res, dict):
                    decision = res.get("decision", "HOLD").upper()
                    tokens_in = res.get("tokens_in", 0)
                    tokens_out = res.get("tokens_out", 0)
                    if tokens_in > 0 or tokens_out > 0:
                        self.llm_calls += 1
                elif isinstance(res, tuple):
                    decision = str(res[0]).upper()
                    if len(res) > 1 and isinstance(res[1], dict):
                        tokens_in = res[1].get("tokens_in", 0)
                        tokens_out = res[1].get("tokens_out", 0)
                        self.llm_calls += 1
                else:
                    decision = str(res).upper()
                    # Automated fallback estimation if LLM-mode is enabled
                    if estimate_llm_tokens and len(current_news) > 20:
                        # Split by space is a simple estimation of word-to-token count (approx 1.3 tokens per word)
                        tokens_in = int(len(current_news.split()) * 1.3)
                        tokens_out = 100  # Default assumed output length for decision reasoning
                        self.llm_calls += 1
            except Exception as e:
                # Strategy threw an exception: log failure, treat as HOLD to preserve capital
                print(f"Warning: Strategy execution error on {current_time}: {e}")
                decision = "HOLD"
                
            self.total_tokens_in += tokens_in
            self.total_tokens_out += tokens_out
            
            # Step 2: Execute Decisions & Apply Commissions
            portfolio_value = self.cash + (self.shares * current_price)
            executed_trade = False
            fee = 0.0
            
            if decision == "BUY" and self.cash > 1.0:
                # Reinvest all remaining cash into SPY, deducting fee
                gross_amount = self.cash
                fee = gross_amount * self.commission
                net_amount = gross_amount - fee
                shares_bought = net_amount / current_price
                
                self.shares += shares_bought
                self.cash = 0.0
                executed_trade = True
                
            elif decision == "SELL" and self.shares > 0.0:
                # Sell all SPY shares, deducting fee
                gross_amount = self.shares * current_price
                fee = gross_amount * self.commission
                net_cash = gross_amount - fee
                
                self.cash += net_cash
                self.shares = 0.0
                executed_trade = True
                
            # Step 3: Log state
            new_portfolio_value = self.cash + (self.shares * current_price)
            self.portfolio_value = new_portfolio_value
            portfolio_history.append(self.portfolio_value)
            
            if executed_trade:
                trade_log.append({
                    "date": current_time.strftime("%Y-%m-%d"),
                    "action": decision,
                    "price": round(current_price, 4),
                    "fee": round(fee, 4),
                    "cash": round(self.cash, 2),
                    "shares": round(self.shares, 4),
                    "portfolio_value": round(self.portfolio_value, 2)
                })

        # Calculate Hardware Power Telemetry
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        
        # Calculate Energy
        end_cpu_joules = get_cpu_energy_joules()
        measured_energy = False
        total_joules = 0.0
        
        if start_cpu_joules is not None and end_cpu_joules is not None:
            # CPU sensor worked
            total_joules = end_cpu_joules - start_cpu_joules
            measured_energy = True
            # Check for GPU power if GPU mode active
            if model_tdp_mode == "gpu":
                gpu_watts = get_gpu_power_watts()
                if gpu_watts is not None:
                    total_joules += (gpu_watts * elapsed_seconds)
        else:
            # Fallback estimation using TDP
            tdp = DEFAULT_GPU_TDP if model_tdp_mode == "gpu" else DEFAULT_CPU_TDP
            total_joules = elapsed_seconds * tdp
            
        return self.compile_metrics(portfolio_history, trade_log, elapsed_seconds, total_joules, measured_energy)

    def compile_metrics(self, history, trade_log, elapsed_time, joules, measured_energy):
        """Compiles backtesting statistics, P&L, token pricing, and final Net AROI score."""
        initial_val = self.initial_cash
        final_val = self.portfolio_value
        
        # Standard Financial Returns
        total_return_pct = ((final_val - initial_val) / initial_val) * 100.0
        
        # Benchmark Return (Buy & Hold SPY)
        bench_start = self.df['Close'].iloc[0]
        bench_end = self.df['Close'].iloc[-1]
        benchmark_return_pct = ((bench_end - bench_start) / bench_start) * 100.0
        
        # Max Drawdown
        history_arr = np.array(history)
        peaks = np.maximum.accumulate(history_arr)
        drawdowns = (history_arr - peaks) / peaks
        max_drawdown = float(drawdowns.min() * 100.0) if len(drawdowns) > 0 else 0.0
        
        # Daily Returns & Annualized Sharpe Ratio
        daily_returns = np.diff(history_arr) / history_arr[:-1] if len(history_arr) > 1 else np.array([])
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            # Assuming 252 trading days per year
            sharpe_ratio = float(np.sqrt(252) * (daily_returns.mean() / daily_returns.std()))
        else:
            sharpe_ratio = 0.0
            
        # Win Rate
        wins = 0
        losses = 0
        # Analyze pairs of trades (BUY followed by SELL) to calculate trading win rate
        buys = [t for t in trade_log if t['action'] == 'BUY']
        sells = [t for t in trade_log if t['action'] == 'SELL']
        
        for b, s in zip(buys, sells):
            # Win if sell portfolio value was higher than buy portfolio value
            if s['portfolio_value'] > b['portfolio_value']:
                wins += 1
            else:
                losses += 1
        total_closed_trades = wins + losses
        win_rate = (wins / total_closed_trades * 100.0) if total_closed_trades > 0 else 0.0
        
        # Compute-to-Revenue Pricing calculation
        token_cost = (self.total_tokens_in / 1000000.0 * PRICE_PER_M_INPUT_TOKENS) + \
                     (self.total_tokens_out / 1000000.0 * PRICE_PER_M_OUTPUT_TOKENS)
                     
        kilojoules = joules / 1000.0
        energy_cost = kilojoules * CARBON_COST_PER_KJ
        
        total_compute_cost = token_cost + energy_cost
        net_profit_usd = final_val - initial_val
        
        # Compute final Net Agentic ROI (AROI)
        # Score = Net Profit / Compute Cost
        # To avoid division by zero when compute cost is extremely small or zero (like rule-based),
        # we add a tiny floor of $0.0001 (representing the micro cost of CPU time)
        compute_cost_denominator = max(total_compute_cost, 0.0001)
        aroi_score = net_profit_usd / compute_cost_denominator
        
        results = {
            "regime": "Backtest",
            "metrics": {
                "initial_portfolio_value": round(initial_val, 2),
                "final_portfolio_value": round(final_val, 2),
                "total_return_pct": round(total_return_pct, 4),
                "benchmark_return_pct": round(benchmark_return_pct, 4),
                "alpha_vs_benchmark": round(total_return_pct - benchmark_return_pct, 4),
                "max_drawdown_pct": round(max_drawdown, 4),
                "sharpe_ratio_annualized": round(sharpe_ratio, 4),
                "win_rate_pct": round(win_rate, 2),
                "total_trades_executed": len(trade_log),
                "total_closed_trades": total_closed_trades
            },
            "telemetry": {
                "elapsed_execution_seconds": round(elapsed_time, 4),
                "total_energy_joules": round(joules, 2),
                "measured_via_hardware_sensors": measured_energy,
                "input_tokens_consumed": self.total_tokens_in,
                "output_tokens_consumed": self.total_tokens_out,
                "total_llm_calls": self.llm_calls,
                "calculated_costs": {
                    "token_cost_usd": round(token_cost, 6),
                    "energy_cost_usd": round(energy_cost, 6),
                    "total_compute_cost_usd": round(total_compute_cost, 6)
                }
            },
            "scores": {
                "net_profit_loss_usd": round(net_profit_usd, 2),
                "efficiency_score_aroi": round(aroi_score, 4)
            },
            "trade_log": trade_log
        }
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Agentic System Backtesting Engine")
    parser.add_argument("--data-file", type=str, required=True, help="Path to simulation pack CSV (filtered/unfiltered)")
    parser.add_argument("--initial-cash", type=float, default=10000.0, help="Initial portfolio cash in USD (default: 10000.0)")
    parser.add_argument("--commission", type=float, default=0.0005, help="Commission rate per trade (default: 0.0005, i.e. 0.05%)")
    parser.add_argument("--strategy", type=str, default="strategies.strategy_logic", help="Module path for the strategy to load (default: strategies.strategy_logic)")
    parser.add_argument("--estimate-tokens", action="store_true", help="Automatically estimate input/output tokens from news context size if strategy uses LLMs")
    parser.add_argument("--hardware-mode", type=str, default="cpu", choices=["cpu", "gpu"], help="Target hardware execution tracking profile (default: cpu)")
    parser.add_argument("--output", type=str, default="telemetry/results.json", help="Path to save simulation telemetry results (default: telemetry/results.json)")
    
    args = parser.parse_args()
    
    try:
        engine = BacktestEngine(
            data_path=args.data_file,
            initial_cash=args.initial_cash,
            commission=args.commission,
            strategy_module=args.strategy
        )
        
        results = engine.run(
            estimate_llm_tokens=args.estimate_tokens,
            model_tdp_mode=args.hardware_mode
        )
        
        # Save output
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\n✅ Backtest completed successfully! Telemetry results written to: {args.output}")
        print(f"📊 Total Return: {results['metrics']['total_return_pct']}% (Benchmark: {results['metrics']['benchmark_return_pct']}%)")
        print(f"💰 Net P&L: ${results['scores']['net_profit_loss_usd']} | Net AROI Score: {results['scores']['efficiency_score_aroi']}")
        
    except Exception as e:
        print(f"❌ Error during backtest execution: {e}")
        # Write structural failure log
        fail_results = {
            "status": "FAILED",
            "error": str(e),
            "scores": {"efficiency_score_aroi": -100.0, "net_profit_loss_usd": 0.0}
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(fail_results, f, indent=4)
        sys.exit(1)
