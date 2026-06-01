#!/usr/bin/env python3
# ==============================================================================
# Recursive Quantitative Trading Orchestrator
# Location: /home/ow9800/recursive-financial-agents/engine/orchestrator.py
# ==============================================================================
# This module acts as the "Nerve Center" orchestrator. It manages the recursive
# self-improvement loop: backtesting strategy code, analyzing performance,
# storing logs and semantic lessons in PostgreSQL LTM, prompting the Gemini
# Architect with quantitative & qualitative feedback, and committing code changes
# to evolutionary git branches/PRs.

import os
import sys
import time
import json
import hashlib
import subprocess
import re
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.api_connectors import APIConnector, load_env_variables
from engine.backtest_engine import BacktestEngine

# Ensure env variables are loaded
load_env_variables()


# ==============================================================================
# Git and Shell Utilities
# ==============================================================================
def run_cmd(cmd, cwd=None, check=True):
    """Executes a system shell command safely."""
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True
        )
        if check and res.returncode != 0:
            raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr.strip()}")
        return res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        if check:
            raise e
        return "", str(e)


def init_git_repo():
    """Checks git configuration and confirms the user identity is set."""
    try:
        run_cmd("git config --global user.email")
    except Exception:
        print("Configuring fallback Git user identity...")
        run_cmd('git config --global user.email "ow9800@recursive-trader.ai"')
        run_cmd('git config --global user.name "Recursive Agent"')


# ==============================================================================
# Database Helper Utilities
# ==============================================================================
class DatabaseManager:
    def __init__(self, dbname="recursive_trading", host="/var/run/postgresql"):
        self.dbname = dbname
        self.host = host
        self.conn = None
        self.connect()

    def connect(self):
        """Creates a Unix socket connection to local PostgreSQL."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                host=self.host,
                user="ow9800"
            )
            self.conn.autocommit = True
        except Exception as e:
            print(f"CRITICAL: Failed to connect to PostgreSQL database: {e}")
            raise e

    def _sanitize(self, obj):
        """Recursively converts NumPy scalars and objects to standard Python types."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(x) for x in obj]
        elif hasattr(obj, "item") and callable(getattr(obj, "item", None)):
            return obj.item()
        else:
            return obj

    def close(self):
        if self.conn:
            self.conn.close()

    def get_strategy_by_id(self, strategy_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM strategies WHERE strategy_id = %s", (strategy_id,))
            return cur.fetchone()

    def get_baseline_strategy(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM strategies WHERE generation = 0 ORDER BY created_at DESC LIMIT 1")
            return cur.fetchone()

    def save_strategy(self, generation, branch_name, code_content, rationale, parent_strategy_id=None):
        code_hash = hashlib.sha256(code_content.encode('utf-8')).hexdigest()
        
        # Check if identical hash already exists to prevent duplicate strategies
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT strategy_id FROM strategies WHERE code_hash = %s", (code_hash,))
            existing = cur.fetchone()
            if existing:
                print(f"Strategy code identical to existing strategy: {existing['strategy_id']}")
                return existing['strategy_id']

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO strategies (generation, branch_name, code_content, code_hash, rationale, parent_strategy_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING strategy_id;
                """,
                (generation, branch_name, code_content, code_hash, rationale, parent_strategy_id)
            )
            res = cur.fetchone()
            return res['strategy_id']

    def save_run(self, strategy_id, regime, metrics, telemetry, scores, trade_log):
        # Sanitize any NumPy data types to standard Python types
        metrics = self._sanitize(metrics)
        telemetry = self._sanitize(telemetry)
        scores = self._sanitize(scores)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    strategy_id, regime, initial_portfolio_value, final_portfolio_value, 
                    total_return_pct, benchmark_return_pct, alpha_vs_benchmark, max_drawdown_pct, 
                    sharpe_ratio_annualized, win_rate_pct, total_trades_executed, total_closed_trades, 
                    elapsed_seconds, total_energy_joules, measured_via_hardware, 
                    input_tokens_consumed, output_tokens_consumed, total_llm_calls, 
                    token_cost_usd, energy_cost_usd, total_compute_cost_usd, 
                    net_profit_loss_usd, efficiency_score_aroi, trade_log_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING run_id;
                """,
                (
                    strategy_id,
                    regime,
                    metrics["initial_portfolio_value"],
                    metrics["final_portfolio_value"],
                    metrics["total_return_pct"],
                    metrics["benchmark_return_pct"],
                    metrics["alpha_vs_benchmark"],
                    metrics["max_drawdown_pct"],
                    metrics["sharpe_ratio_annualized"],
                    metrics["win_rate_pct"],
                    metrics["total_trades_executed"],
                    metrics["total_closed_trades"],
                    telemetry["elapsed_execution_seconds"],
                    telemetry["total_energy_joules"],
                    telemetry["measured_via_hardware_sensors"],
                    telemetry["input_tokens_consumed"],
                    telemetry["output_tokens_consumed"],
                    telemetry["total_llm_calls"],
                    telemetry["calculated_costs"]["token_cost_usd"],
                    telemetry["calculated_costs"]["energy_cost_usd"],
                    telemetry["calculated_costs"]["total_compute_cost_usd"],
                    scores["net_profit_loss_usd"],
                    scores["efficiency_score_aroi"],
                    json.dumps(trade_log)
                )
            )
            res = cur.fetchone()
            return res['run_id']

    def save_lesson(self, strategy_id, run_id, summary, sentiment_pattern, key_failure_cause, embedding_vector):
        # Convert vector to Postgres string format '[v1, v2, v3...]'
        vector_str = f"[{','.join(map(str, embedding_vector))}]" if embedding_vector else None
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO lessons_learned (strategy_id, run_id, summary, sentiment_pattern, key_failure_cause, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING lesson_id;
                """,
                (strategy_id, run_id, summary, sentiment_pattern, key_failure_cause, vector_str)
            )
            res = cur.fetchone()
            return res['lesson_id']

    def get_best_strategies(self, limit=3):
        """Retrieves top strategies sorted by efficiency score (AROI) or total return."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.strategy_id, s.generation, s.branch_name, s.code_content, r.total_return_pct, r.efficiency_score_aroi 
                FROM strategies s
                JOIN runs r ON s.strategy_id = r.strategy_id
                WHERE r.regime = 'Normal_Filtered'
                ORDER BY r.efficiency_score_aroi DESC, r.total_return_pct DESC
                LIMIT %s;
                """,
                (limit,)
            )
            return cur.fetchall()

    def get_similar_lessons(self, embedding_vector, limit=3):
        """Retrieves semantically similar failure causes or successful patterns via pgvector cosine distance."""
        if not embedding_vector:
            # Fallback to recent lessons if embedding failed
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT summary, sentiment_pattern, key_failure_cause FROM lessons_learned ORDER BY created_at DESC LIMIT %s", (limit,))
                return cur.fetchall()
                
        vector_str = f"[{','.join(map(str, embedding_vector))}]"
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT summary, sentiment_pattern, key_failure_cause, (embedding <=> %s::vector) as distance
                FROM lessons_learned
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (vector_str, limit)
            )
            return cur.fetchall()


# ==============================================================================
# LLM Response Parsers
# ==============================================================================
def parse_architect_response(response_text):
    """
    Parses Gemini's code generation response.
    Expects python code enclosed in ```python ... ``` and reasoning in <RATIONALE> ... </RATIONALE> blocks.
    """
    # 1. Extract rationale
    rationale = "Self-improved evolutionary strategy"
    rationale_match = re.search(r"<RATIONALE>(.*?)</RATIONALE>", response_text, re.DOTALL | re.IGNORECASE)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    else:
        # Fallback to looking for text before code block
        text_before_code = response_text.split("```")[0].strip()
        if text_before_code:
            rationale = text_before_code[:500]

    # 2. Extract python code
    code_content = None
    # Support case-insensitive 'python', 'python3', or 'py' with optional digits
    code_match = re.search(r"```(?:python|py)\d*\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if code_match:
        code_content = code_match.group(1).strip()
    else:
        # Match any generic code block if python specific wasn't found
        code_match_generic = re.search(r"```\w*\s*(.*?)\s*```", response_text, re.DOTALL)
        if code_match_generic:
            code_content = code_match_generic.group(1).strip()
        elif "def decide" in response_text:
            # If no block at all, but looks like valid python script
            code_content = response_text.strip()

    return code_content, rationale


def parse_analyzer_response(response_text):
    """
    Parses Gemini's post-mortem analysis response.
    Expects output structure with XML-style brackets.
    """
    summary = "Strategy completed backtest."
    sentiment_pattern = "N/A"
    key_failure_cause = "N/A"

    summary_match = re.search(r"<SUMMARY>(.*?)</SUMMARY>", response_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        summary = response_text.split("\n")[0][:300]

    sentiment_match = re.search(r"<SENTIMENT_PATTERN>(.*?)</SENTIMENT_PATTERN>", response_text, re.DOTALL | re.IGNORECASE)
    if sentiment_match:
        sentiment_pattern = sentiment_match.group(1).strip()

    failure_match = re.search(r"<FAILURE_CAUSE>(.*?)</FAILURE_CAUSE>", response_text, re.DOTALL | re.IGNORECASE)
    if failure_match:
        key_failure_cause = failure_match.group(1).strip()

    return summary, sentiment_pattern, key_failure_cause


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recursive Self-Improving Trading Orchestrator")
    parser.add_argument("--generations", type=int, default=10, help="Number of evolutionary generations to execute")
    parser.add_argument("--dataset-normal", type=str, default="data/simulation_pack_filtered_normal.csv", help="High-signal normal dataset")
    parser.add_argument("--dataset-stress", type=str, default="data/simulation_pack_filtered.csv", help="Covid filtered stress-testing dataset")
    parser.add_argument("--architect-model", type=str, default="gemini-2.5-flash", help="Model used to generate strategy code (gemini-2.5-flash or gemini-2.5-pro)")
    parser.add_argument("--pacing-delay", type=float, default=4.0, help="Physical API pacing delay in seconds")
    parser.add_argument("--skip-pr", action="store_true", help="Skip creating remote PRs via gh CLI")
    
    args = parser.parse_args()

    print("\n" + "="*80)
    print("      LAUNCHING RECURSIVE SELF-IMPROVING QUANTITATIVE TRADING LOOP")
    print("="*80)
    print(f"Target Generations: {args.generations}")
    print(f"Normal Dataset:     {args.dataset_normal}")
    print(f"Stress Dataset:     {args.dataset_stress}")
    print(f"Architect Model:    {args.architect_model}")
    print(f"API Pacing Delay:   {args.pacing_delay}s")
    print("="*80 + "\n")

    # Initialize Clients
    db = DatabaseManager()
    init_git_repo()
    
    # We use Google AI Studio for the Architect and Analyzer
    api_google = APIConnector(provider="google", model=args.architect_model, pacing_delay=args.pacing_delay)
    
    # --------------------------------------------------------------------------
    # STEP 1: LOAD & RUN BASELINE STRATEGY (GENERATION 0)
    # --------------------------------------------------------------------------
    print(">>> Initializing Baseline Strategy (Generation 0)...")
    
    # Read current strategy code
    strategy_path = "strategies/strategy_logic.py"
    with open(strategy_path, "r") as f:
        baseline_code = f.read()

    # Query if baseline already recorded
    baseline_db = db.get_baseline_strategy()
    if baseline_db:
        print(f"Baseline strategy already recorded with ID: {baseline_db['strategy_id']}. Resuming loop.")
        baseline_id = baseline_db['strategy_id']
    else:
        # Save baseline strategy
        baseline_id = db.save_strategy(
            generation=0,
            branch_name="main",
            code_content=baseline_code,
            rationale="Initial human-written rule-based baseline strategy.",
            parent_strategy_id=None
        )
        print(f"Recorded Baseline Strategy to PostgreSQL with ID: {baseline_id}")

        # Run Backtest on Normal Dataset
        print("Running normal regime backtest for Baseline...")
        try:
            engine_normal = BacktestEngine(args.dataset_normal, strategy_module="strategies.strategy_logic")
            results_normal = engine_normal.run(estimate_llm_tokens=False, plot_path="telemetry/gen_0_normal_equity.png")
            
            # Save Run Results
            run_normal_id = db.save_run(
                strategy_id=baseline_id,
                regime="Normal_Filtered",
                metrics=results_normal["metrics"],
                telemetry=results_normal["telemetry"],
                scores=results_normal["scores"],
                trade_log=results_normal["trade_log"]
            )
            print(f"Saved Gen 0 Normal Run: Return = {results_normal['metrics']['total_return_pct']:.2f}% | AROI = {results_normal['scores']['efficiency_score_aroi']:.2f}")

            # Run Backtest on Stress Dataset (Covid)
            print("Running stress regime backtest for Baseline...")
            engine_stress = BacktestEngine(args.dataset_stress, strategy_module="strategies.strategy_logic")
            results_stress = engine_stress.run(estimate_llm_tokens=False, plot_path="telemetry/gen_0_stress_equity.png")
            
            run_stress_id = db.save_run(
                strategy_id=baseline_id,
                regime="Covid_Filtered",
                metrics=results_stress["metrics"],
                telemetry=results_stress["telemetry"],
                scores=results_stress["scores"],
                trade_log=results_stress["trade_log"]
            )
            print(f"Saved Gen 0 Stress Run: Return = {results_stress['metrics']['total_return_pct']:.2f}% | AROI = {results_stress['scores']['efficiency_score_aroi']:.2f}")

            # Generate semantic qualitative analysis for baseline
            print("Generating qualitative analysis for Baseline...")
            post_mortem_prompt = f"""
            Analyze the following backtesting performance for SPY.
            
            NORMAL MARKET REGIME:
            - Return: {results_normal['metrics']['total_return_pct']}%
            - Max Drawdown: {results_normal['metrics']['max_drawdown_pct']}%
            - Sharpe Ratio: {results_normal['metrics']['sharpe_ratio_annualized']}
            - Win Rate: {results_normal['metrics']['win_rate_pct']}%
            - Total Closed Trades: {results_normal['metrics']['total_closed_trades']}
            
            COVID STRESS REGIME:
            - Return: {results_stress['metrics']['total_return_pct']}%
            - Max Drawdown: {results_stress['metrics']['max_drawdown_pct']}%
            - Sharpe Ratio: {results_stress['metrics']['sharpe_ratio_annualized']}
            - Win Rate: {results_stress['metrics']['win_rate_pct']}%
            - Total Closed Trades: {results_stress['metrics']['total_closed_trades']}

            Provide a qualitative analysis using exactly this format:
            <SUMMARY>A concise qualitative summary of the strategy's performance characteristics across both regimes.</SUMMARY>
            <SENTIMENT_PATTERN>Identify sentiment words or headlines that triggered false-positives or highly profitable trades.</SENTIMENT_PATTERN>
            <FAILURE_CAUSE>Identify why the strategy lost money or incurred large drawdowns (e.g., look-ahead bias, lagging indicators, false sentiment triggers).</FAILURE_CAUSE>
            """
            analysis_res = api_google.query(post_mortem_prompt)
            summary, sentiment_pattern, key_failure_cause = parse_analyzer_response(analysis_res["text"])
            
            # Embed the qualitative post-mortem
            lesson_text = f"Summary: {summary}. Sentiment: {sentiment_pattern}. Failures: {key_failure_cause}."
            embedding_vector = api_google.embed(lesson_text)
            
            # Save Lesson
            db.save_lesson(baseline_id, run_normal_id, summary, sentiment_pattern, key_failure_cause, embedding_vector)
            print("Baseline lesson successfully embedded and logged to LTM database.")
        except Exception as e:
            print(f"⚠️ Warning: Gen 0 backtest or logging encountered an issue: {e}")

    # Set initial parent pointers and start generation
    start_gen = 1
    parent_id = baseline_id
    parent_code = baseline_code
    best_normal_aroi = -99999.0

    # Query if there are any already evolved strategies with runs in the DB
    try:
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.strategy_id, s.generation, s.code_content, r.efficiency_score_aroi 
                FROM strategies s
                JOIN runs r ON s.strategy_id = r.strategy_id
                WHERE r.regime = 'Normal_Filtered'
                ORDER BY s.generation DESC, r.efficiency_score_aroi DESC
                LIMIT 1;
                """
            )
            latest_run = cur.fetchone()
            if latest_run and latest_run['generation'] >= 1:
                start_gen = latest_run['generation'] + 1
                parent_id = latest_run['strategy_id']
                parent_code = latest_run['code_content']
                best_normal_aroi = float(latest_run['efficiency_score_aroi'])
                print(f">>> RESUMING LOOP: Found existing evolved strategy at Gen {latest_run['generation']} (ID: {parent_id[:8]}) with AROI: {best_normal_aroi:.4f}")
            else:
                # Fallback to baseline (Gen 0) run query if no evolved run exists
                cur.execute("SELECT efficiency_score_aroi FROM runs WHERE strategy_id = %s AND regime = 'Normal_Filtered' LIMIT 1", (parent_id,))
                run_row = cur.fetchone()
                if run_row:
                    best_normal_aroi = float(run_row['efficiency_score_aroi'])
                print(f">>> STARTING NEW LOOP: Baseline parent established with AROI: {best_normal_aroi:.4f}")
    except Exception as e:
        print(f"⚠️ Warning during resume query: {e}. Defaulting to starting loop from scratch.")

    # Write current parent code back to active strategy logic to ensure consistency
    try:
        with open(strategy_path, "w") as f:
            f.write(parent_code)
        print(f"Verified and synchronized '{strategy_path}' with parent strategy code.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to write parent code to active logic: {e}")

    # --------------------------------------------------------------------------
    # STEP 2: EVOLUTIONARY RECURSIVE LOOP (GENERATIONS 1 to N)
    # --------------------------------------------------------------------------
    for g in range(start_gen, args.generations + 1):
        print("\n" + "="*80)
        print(f">>> STARTING EVOLUTION GENERATION {g} of {args.generations} <<<")
        print("="*80)
        
        # 1. Retrieve Past Lessons & Code Examples from LTM database
        print("Retrieving past semantic lessons & high-performing strategies from LTM...")
        # Get the top best performing strategy to show the LLM what has worked (limited to 1 to save prompt token budget)
        best_strats = db.get_best_strategies(limit=1)
        best_examples_str = ""
        for i, bs in enumerate(best_strats):
            best_examples_str += f"\n--- PAST BEST STRATEGY #{i+1} (AROI: {bs['efficiency_score_aroi']:.2f}, Return: {bs['total_return_pct']:.2f}%) ---\n"
            # Show only relevant structure to save tokens if code is long, or show complete code
            best_examples_str += bs['code_content'] + "\n"

        # Embed current performance profile of the active parent to query matching failure modes
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT summary, key_failure_cause FROM lessons_learned 
                WHERE strategy_id = %s LIMIT 1
                """, 
                (parent_id,)
            )
            parent_lesson = cur.fetchone()
            
        semantic_retrieved_lessons = ""
        if parent_lesson:
            parent_failure_text = parent_lesson.get('key_failure_cause', '') or ""
            if parent_failure_text:
                print(f"Generating query embedding for active failure description: '{parent_failure_text[:60]}...'")
                try:
                    parent_embed = api_google.embed(parent_failure_text)
                    similar_lessons = db.get_similar_lessons(parent_embed, limit=3)
                    for sl in similar_lessons:
                        semantic_retrieved_lessons += f"- Failure Cause: {sl['key_failure_cause']}\n  Sentiment Patterns observed: {sl['sentiment_pattern']}\n  Performance Summary: {sl['summary']}\n\n"
                except Exception as e:
                    print(f"Warning: Failed to search semantic lessons: {e}")

        # If empty, do database sorted retrieval
        if not semantic_retrieved_lessons:
            with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT summary, sentiment_pattern, key_failure_cause FROM lessons_learned ORDER BY created_at DESC LIMIT 3")
                for sl in cur.fetchall():
                    semantic_retrieved_lessons += f"- Failure Cause: {sl['key_failure_cause']}\n  Sentiment Patterns: {sl['sentiment_pattern']}\n  Summary: {sl['summary']}\n\n"

        # 2. Prompt the Architect (Gemini 2.5 Flash / Pro) to Evolve the Strategy
        print("Formulating evolution prompt for the Gemini Architect...")
        
        architect_prompt = f"""
You are the Lead Quantitative Architect in a Recursive Self-Improving Trading System.
Your task is to analyze the active S&P 500 ($SPY) trading strategy, reflect on quantitative and qualitative feedback, review lessons from past runs, and output a SELF-IMPROVED version of the trading code.

YOUR ABSOLUTE CONSTRAINTS:
1. You must output a valid Python module containing a `decide(current_price, price_history, news_context)` function.
2. The function parameters must be EXACTLY:
   - `current_price` (float): The current day's closing price for SPY.
   - `price_history` (list of float): Daily historical closing prices up to yesterday. (Can be empty on day 0).
   - `news_context` (str): Combined news headlines from the last 24 hours.
3. The return value of `decide()` must be EXACTLY:
   - "BUY" (to invest all cash in SPY),
   - "SELL" (to liquidate all SPY shares back to cash), or
   - "HOLD" (to keep the active state).
4. No look-ahead bias is allowed. You must only utilize historical price_history and today's news_context.
5. All external libraries must be robustly imported. Stick to standard packages like `pandas`, `numpy`, or basic Python types. Do not use complex external data APIs.

--- ACTIVE PARENT STRATEGY CODE ---
```python
{parent_code}
```

--- RECENT QUANTITATIVE METRICS ---
Active Parent AROI (Agentic Return on Investment): {best_normal_aroi:.4f}

--- RETRIEVED SEMANTIC LESSONS (LTM DATABASE) ---
The following are critical failure modes and lessons learned from similar past runs:
{semantic_retrieved_lessons}

--- SEED SUCCESS PATTERNS (HISTORICAL WINNERS) ---
Review what has worked well in the past to inspire indicator logic:
{best_examples_str}

Reflect on how to optimize the active code:
- Can you introduce better technical indicators? (e.g., Exponential Moving Averages (EMA), MACD-like trends, RSI, or ATR-based volatility filters)?
- How can you make sentiment-keyword matching more robust? (e.g., scoring phrases, utilizing weightings, adding negative negation filters, or handling word boundary checks)?
- Can you dynamically scale indicators based on price history length?
- Focus on maximizing the AROI: avoiding false buy/sell signals, reducing transaction fees by filtering noise, and surviving stress regimes (like Covid) while maintaining performance.

Provide your response in exactly this format. Make sure the rationale is wrapped inside <RATIONALE> and the complete modified Python code is enclosed in ```python ... ``` blocks.

CRITICAL INSTRUCTION: To prevent response truncation, keep your rationale inside <RATIONALE> extremely concise (under 100 words). Do not write a long verbose explanation, and do not duplicate code within your rationale. Prioritize outputting the complete, self-contained Python code in the ```python block.

<RATIONALE>
Provide a very short summary (under 100 words) of your design decisions, indicators added/updated, and how past failure lessons were addressed.
</RATIONALE>

```python
# Complete self-contained python code here with imports and the decide() function.
```
"""
        
        print(f"Calling Architect API ({api_google.model})...")
        try:
            architect_res = api_google.query(architect_prompt, temperature=0.3)
            evolved_code, rationale = parse_architect_response(architect_res["text"])
            
            if not evolved_code:
                print("❌ Error: Gemini did not return a valid code block. Skipping generation.")
                print(f"--- DEBUG: RAW MODEL RESPONSE (First 4000 chars) ---\n{architect_res['text'][:4000]}\n" + "="*40)
                continue
                
            print(f"\n--- Architect Rationale ---\n{rationale}\n" + "-"*40)
        except Exception as e:
            print(f"❌ Error during Architect API call: {e}")
            continue

        # 3. Code Validation and Sandboxing
        print("Validating evolved strategy syntax and safety...")
        temp_module_path = "strategies/temp_evolution_logic.py"
        try:
            with open(temp_module_path, "w") as f:
                f.write(evolved_code)
                
            # Verify dynamic import compilation
            if "strategies.temp_evolution_logic" in sys.modules:
                importlib = sys.modules["importlib"]
                importlib.reload(sys.modules["strategies.temp_evolution_logic"])
            else:
                __import__("strategies.temp_evolution_logic")
                
            # Basic functional test
            temp_decide = sys.modules["strategies.temp_evolution_logic"].decide
            test_decision = temp_decide(400.0, [395.0, 396.0, 398.0, 399.0], "Federal Reserve hints at future interest rate cut, markets rally.")
            assert test_decision in ["BUY", "SELL", "HOLD"], "decide() returned an invalid action!"
            print(f"✅ Code compiled and functional test passed! Sample decision: {test_decision}")
        except Exception as e:
            print(f"❌ Structural code validation failed: {e}. Skipping this generation.")
            if os.path.exists(temp_module_path):
                os.remove(temp_module_path)
            continue

        # 4. Git Branching & Active Code Deployment
        branch_name = f"evolution_gen_{g}"
        print(f"Committing changes to branch: {branch_name}...")
        try:
            # Checkout a clean branch
            run_cmd(f"git checkout -b {branch_name}", check=False) # might fail if already exists, that's okay
            run_cmd(f"git checkout {branch_name}")
            
            # Write evolved code to active logic file
            with open(strategy_path, "w") as f:
                f.write(evolved_code)
                
            # Clean up temp file
            if os.path.exists(temp_module_path):
                os.remove(temp_module_path)
                
            # Commit local code
            run_cmd(f"git add {strategy_path}")
            run_cmd(f'git commit -m "Evolve strategy generation {g} - rationale: {rationale[:100].replace(chr(34), chr(39))}"', check=False)
            print(f"Git commit created for generation {g}.")
        except Exception as e:
            print(f"⚠️ Git operations failed: {e}. Running local backtest in active workspace anyway.")

        # 5. Execute Backtest Sandboxing
        print(f"Running backtest sandboxing for Gen {g}...")
        try:
            # Run on normal regime
            engine_normal = BacktestEngine(args.dataset_normal, strategy_module="strategies.strategy_logic")
            results_normal = engine_normal.run(estimate_llm_tokens=False, plot_path=f"telemetry/gen_{g}_normal_equity.png")
            
            # Save strategy representation to db
            strategy_id = db.save_strategy(
                generation=g,
                branch_name=branch_name,
                code_content=evolved_code,
                rationale=rationale,
                parent_strategy_id=parent_id
            )
            
            # Save normal run
            run_normal_id = db.save_run(
                strategy_id=strategy_id,
                regime="Normal_Filtered",
                metrics=results_normal["metrics"],
                telemetry=results_normal["telemetry"],
                scores=results_normal["scores"],
                trade_log=results_normal["trade_log"]
            )
            print(f"Gen {g} Normal: Return = {results_normal['metrics']['total_return_pct']:.2f}% | AROI = {results_normal['scores']['efficiency_score_aroi']:.2f}")

            # Run on stress regime (Covid)
            engine_stress = BacktestEngine(args.dataset_stress, strategy_module="strategies.strategy_logic")
            results_stress = engine_stress.run(estimate_llm_tokens=False, plot_path=f"telemetry/gen_{g}_stress_equity.png")
            
            # Save stress run
            run_stress_id = db.save_run(
                strategy_id=strategy_id,
                regime="Covid_Filtered",
                metrics=results_stress["metrics"],
                telemetry=results_stress["telemetry"],
                scores=results_stress["scores"],
                trade_log=results_stress["trade_log"]
            )
            print(f"Gen {g} Stress: Return = {results_stress['metrics']['total_return_pct']:.2f}% | AROI = {results_stress['scores']['efficiency_score_aroi']:.2f}")

            # 6. Execute Analyzer for Qualitative Post-Mortem & Embedding
            print(f"Analyzing and embedding performance post-mortem for Gen {g}...")
            post_mortem_prompt = f"""
            Analyze the performance of this evolved strategy.
            
            STRATEGY CODE:
            ```python
            {evolved_code}
            ```
            
            NORMAL REGIME BACKTEST RESULTS:
            - Return: {results_normal['metrics']['total_return_pct']}%
            - Max Drawdown: {results_normal['metrics']['max_drawdown_pct']}%
            - Sharpe Ratio: {results_normal['metrics']['sharpe_ratio_annualized']}
            - Win Rate: {results_normal['metrics']['win_rate_pct']}%
            - Total Closed Trades: {results_normal['metrics']['total_closed_trades']}
            
            COVID STRESS REGIME BACKTEST RESULTS:
            - Return: {results_stress['metrics']['total_return_pct']}%
            - Max Drawdown: {results_stress['metrics']['max_drawdown_pct']}%
            - Sharpe Ratio: {results_stress['metrics']['sharpe_ratio_annualized']}
            - Win Rate: {results_stress['metrics']['win_rate_pct']}%
            - Total Closed Trades: {results_stress['metrics']['total_closed_trades']}

            Provide exactly the qualitative analysis inside:
            <SUMMARY>Overall analysis of how this code performed under both market conditions.</SUMMARY>
            <SENTIMENT_PATTERN>Which keywords or news sentiment phrases caused strong trades or false traps?</SENTIMENT_PATTERN>
            <FAILURE_CAUSE>Why did the strategy underperform or encounter drawdowns? If it outperformed, why did it work?</FAILURE_CAUSE>
            """
            analysis_res = api_google.query(post_mortem_prompt)
            summary, sentiment_pattern, key_failure_cause = parse_analyzer_response(analysis_res["text"])
            
            # Embed the analysis
            lesson_text = f"Summary: {summary}. Sentiment: {sentiment_pattern}. Failures: {key_failure_cause}."
            embedding_vector = api_google.embed(lesson_text)
            
            # Save lesson to LTM DB
            db.save_lesson(strategy_id, run_normal_id, summary, sentiment_pattern, key_failure_cause, embedding_vector)
            print("Logged semantic lesson learned to LTM database.")

            # 7. Monotonic Improvement Guard and Promotion Logic
            new_aroi = float(results_normal["scores"]["efficiency_score_aroi"])
            new_return = float(results_normal["metrics"]["total_return_pct"])
            
            # We promote based on Net Profit or AROI. Let's look at AROI.
            # If the evolved strategy has higher normal regime AROI (or if returns are better with similar costs), promote!
            # To be safe, let's promote if normal AROI is better or return is higher with safe drawdown.
            is_better = False
            if new_aroi > best_normal_aroi:
                is_better = True
            elif new_aroi == best_normal_aroi and new_return > float(db.get_strategy_by_id(parent_id).get('total_return_pct', -999.0) or -999.0):
                is_better = True

            if is_better:
                print(f"🎉 SUCCESS: Gen {g} outperformed previous best parent! AROI increased from {best_normal_aroi:.4f} to {new_aroi:.4f}")
                best_normal_aroi = new_aroi
                parent_id = strategy_id
                parent_code = evolved_code
                
                # Merge into main or establish as new active base branch
                try:
                    run_cmd("git checkout main")
                    run_cmd(f"git merge {branch_name} --no-edit", check=False)
                    print("Merged improvements back to main branch.")
                except Exception as e:
                    print(f"⚠️ Git merge failed: {e}")
                    
                # Optionally push to GitHub and raise PR using gh CLI
                if not args.skip_pr:
                    print("Attempting to push branch and raise a GitHub PR...")
                    try:
                        run_cmd(f"git push origin {branch_name}", check=False)
                        pr_body = f"Recursive self-improvement loop generation {g}.\n\n**Performance metrics:**\n- Normal AROI: {new_aroi:.4f}\n- Normal Return: {results_normal['metrics']['total_return_pct']}%\n- Sharpe: {results_normal['metrics']['sharpe_ratio_annualized']}\n\n**Architect rationale:**\n{rationale}"
                        run_cmd(f'gh pr create --title "Evolution Generation {g} - AROI {new_aroi:.2f}" --body "{pr_body}" --head {branch_name} --base main', check=False)
                        print("GitHub PR successfully opened via gh!")
                    except Exception as e:
                        print(f"⚠️ GitHub PR creation skipped: {e}")
            else:
                print(f"📉 REJECTION: Gen {g} underperformed parent (Gen {g} AROI: {new_aroi:.4f} vs Parent AROI: {best_normal_aroi:.4f})")
                print(f"Reverting active logic workspace to Gen {parent_id[:8]}'s best verified code.")
                
                # Revert workspace file back to parent's code to prevent compounding degradation
                with open(strategy_path, "w") as f:
                    f.write(parent_code)
                    
                # Return local workspace git to main branch
                try:
                    run_cmd("git checkout main")
                except Exception as e:
                    print(f"⚠️ Git checkout main failed: {e}")
                    
        except Exception as e:
            print(f"❌ Error during evolutionary testing of Gen {g}: {e}")
            # Revert to ensure next generation starts clean
            with open(strategy_path, "w") as f:
                f.write(parent_code)
            try:
                run_cmd("git checkout main")
            except Exception:
                pass
            continue

    db.close()
    print("\n" + "="*80)
    print("      RECURSIVE EVOLUTIONARY LOOP COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
