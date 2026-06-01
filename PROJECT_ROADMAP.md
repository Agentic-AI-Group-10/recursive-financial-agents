# PROJECT_ROADMAP.md

## Phase 1: Core Infrastructure & Nerve Center (GCP VM)
*   [x] **GCP Project Setup:** Create a new project in the GCP Console and ensure credits are active.
*   [x] **Provision "Nerve Center" VM:** Setup a Linux (Ubuntu) instance on Compute Engine.
*   [x] **Initialize Database:** Install PostgreSQL and the `pgvector` extension for Long-Term Memory (LTM). *(Schema initialized in [database/schema.sql](file:///home/ow9800/recursive-financial-agents/database/schema.sql))*
*   [ ] **Deploy Telemetry Hub:** Setup a basic dashboard (Streamlit) on the VM to track Efficiency Scores.
*   [x] **Collaboration Setup:** Add your partner to the GCP Project and the GitHub repo.

## Phase 2: Orchestration & API Sourcing (GCP VM & DeepInfra)
*   [x] **Link AI Studio to GCP:** Connect your Gemini API usage to your GCP Project for unified tracking.
*   [ ] **Environment Variables Setup:** Create a `.env` file locally on the VM to store Google AI Studio Keys and DeepInfra API keys securely.
*   [ ] **Setup DeepInfra Credentials:** Register and fund a DeepInfra account to utilize low-cost open-weights models (Llama-3/Mistral) as backtest fallbacks/comparisons.

## Phase 2.5: Data Engineering (Multi-Regime Simulation Packs)
*   [x] **Acquire Datasets:** Download $SPY price data and Kaggle "Daily Financial News" locally.
*   [x] **Build the Joiner:** Write a temporal point-in-time left-joiner with look-ahead guards and timezone normalization (`build_simulation_pack.py`).
*   [x] **Dual Simulation Pack Generation (Normal Regime - 2017-2018):**
    *   [x] **High-Signal Filtered Pack:** Generate `simulation_pack_filtered_normal.csv` (macro-keywords + mega-caps).
    *   [x] **Full Unfiltered Pack:** Generate `simulation_pack_unfiltered_normal.csv` (all headlines, 100 safety cap).
*   [x] **Dual Simulation Pack Generation (Stress Test / Black Swan - 2019-2020):**
    *   [x] **High-Signal Filtered Pack:** Generate `simulation_pack_filtered.csv` (macro-keywords + mega-caps).
    *   [x] **Full Unfiltered Pack:** Generate `simulation_pack_unfiltered.csv` (all headlines, 100 safety cap).

## Phase 2.6: Experimental Design (Multi-Regime Filtered vs. Unfiltered)
*   [ ] **In-Sample Training Trials (Normal Cycle - 2017–2018):** Run recursive self-improvement loops on both the normal filtered and normal unfiltered datasets to evolve strategies under stable conditions.
*   [ ] **Out-of-Sample Stress-Testing (Covid Black Swan - 2019–2020):** Evaluate the best-evolved strategies directly on the highly volatile 2019-2020 crash without further optimization to check regime adaptability.
*   [ ] **Comparative Performance Metrics:** Compare across regimes and datasets:
    *   **Monetary Return:** Cumulative return, Sharpe Ratio, Maximum Drawdown.
    *   **Inference Costs & AROI:** Calculate Net AROI: Net Return % - AI Token & Energy Costs.
    *   **Data Signal Density:** Compare the performance delta between Filtered and Unfiltered contexts to see if noise filtering speeds up self-improvement.

## Phase 3: Local Engine & API Connectors
*   [x] **Build Backtesting Engine:** Implement the core daily simulation engine with point-in-time left-join compliance, transactional fees, full P&L metrics, and hardware/token telemetry tracking (`backtest_engine.py`).
*   [x] **Verify Dynamic Charting:** Ensure visual charts save correctly locally to track strategy equity, trade markers, and drawdowns.
*   [ ] **Build API Connector Class:** Write an API connection helper in the engine that seamlessly routes requests to Google AI Studio (Gemini 1.5 Flash) or DeepInfra (Llama-3-8B) with built-in rate-limiting and token usage reporting.

## Phase 4: Recursive Loop Development (GCP VM Local Execution)
*   [ ] **Build the "Architect" Skill:** In Antigravity, write the logic that queries GCP Postgres LTM for past failures before generating new code.
*   [ ] **Automate PRs:** Ensure Antigravity uses the `gh` CLI to open Pull Requests for every iteration.
*   [ ] **Telemetry Pipeline:** Direct all local backtest run outputs into the PostgreSQL `runs` and `lessons_learned` tables.

## Phase 5: Execution & Evaluation (Comparative Run)
*   [ ] **The Dual 3-Hour Marathons:** Launch the recursive self-improvement loop for both Trial A (filtered) and Trial B (unfiltered) on the GCP VM utilizing the free-tier Gemini 1.5 Flash and DeepInfra fallbacks.
*   [ ] **Post-Mortem & Comparative Analysis:** Generate a report comparing the learning speed, trading performance, and cost-efficiency (AROI) of the self-improved agents under both data regimes.
