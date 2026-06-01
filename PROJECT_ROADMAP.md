# PROJECT_ROADMAP.md

## Phase 1: Core Infrastructure & Nerve Center (GCP)
*   [x] **GCP Project Setup:** Create a new project in the GCP Console and ensure credits are active.
*   [ ] **Provision "Nerve Center" VM:** Setup a Linux (Ubuntu) instance on Compute Engine.
*   [ ] **Initialize Database:** Install PostgreSQL and the `pgvector` extension for Long-Term Memory (LTM).
*   [ ] **Deploy Telemetry Hub:** Setup a basic dashboard (Streamlit) on the VM to track Efficiency Scores.
*   [x] **Collaboration Setup:** Add your partner to the GCP Project
*   [ ] **Collaboration Setup part 2** and the GitHub repo.

## Phase 2: Orchestration & Linking (Antigravity & AI Studio)
*   [x] **Link AI Studio to GCP:** Connect your Gemini API usage to your GCP Project for unified tracking.
*   [ ] **Setup Antigravity "Bridge":** Configure SSH keys so Antigravity can talk to the GCP VM (for DB/Logs) and MIT Supercloud (for execution).
*   [ ] **Environment Variables:** Create a `.env` file locally to store Project IDs and API Keys safely.

## Phase 2.5: Data Engineering (Multi-Regime Simulation Packs)
*   [x] **Acquire Datasets:** Download $SPY price data and Kaggle "Daily Financial News" locally.
*   [x] **Build the Joiner:** Write a temporal point-in-time left-joiner with look-ahead guards and timezone normalization (`build_simulation_pack.py`).
*   [x] **Dual Simulation Pack Generation (Normal Regime - 2017-2018):**
    *   [x] **High-Signal Filtered Pack:** Generate `simulation_pack_filtered_normal.csv` (macro-keywords + mega-caps).
    *   [x] **Full Unfiltered Pack:** Generate `simulation_pack_unfiltered_normal.csv` (all headlines, 100 safety cap).
*   [x] **Dual Simulation Pack Generation (Stress Test / Black Swan - 2019-2020):**
    *   [x] **High-Signal Filtered Pack:** Generate `simulation_pack_filtered.csv` (macro-keywords + mega-caps).
    *   [x] **Full Unfiltered Pack:** Generate `simulation_pack_unfiltered.csv` (all headlines, 100 safety cap).
*   [ ] **Deploy to Foundry:** Upload all four simulation packs to the MIT Supercloud `/data` directory.

## Phase 2.6: Experimental Design (Multi-Regime Filtered vs. Unfiltered)
*   [ ] **In-Sample Training Trials (Normal Cycle - 2017–2018):** Run recursive self-improvement loops on both the normal filtered and normal unfiltered datasets to evolve strategies under stable conditions.
*   [ ] **Out-of-Sample Stress-Testing (Covid Black Swan - 2019–2020):** Evaluate the best-evolved strategies directly on the highly volatile 2019-2020 crash without further optimization to check regime adaptability.
*   [ ] **Comparative Performance Metrics:** Compare across regimes and datasets:
    *   **Monetary Return:** Cumulative return, Sharpe Ratio, Maximum Drawdown.
    *   **Inference Costs & AROI:** Calculate Net AROI: Net Return % - AI Token & Energy Costs.
    *   **Data Signal Density:** Compare the performance delta between Filtered and Unfiltered contexts to see if noise filtering speeds up self-improvement.

## Phase 3: Supercloud Preparation (The Foundry)
*   [x] **Build Backtesting Engine:** Implement the core daily simulation engine with point-in-time left-join compliance, transactional fees, full P&L metrics, and hardware/token telemetry tracking (`backtest_engine.py`).
*   [ ] **Stage Historical Data:** Upload all four simulation pack CSVs to Supercloud `/data`.
*   [ ] **Verify Local LLM:** Identify the path for Llama-3/Mistral on the Supercloud.
*   [ ] **Finalize Engine:** Upload `backtest_engine.py` and `submit_job.sh` to Supercloud.

## Phase 4: Recursive Loop Development
*   [ ] **Build the "Architect" Skill:** In Antigravity, write the logic that queries GCP LTM for past failures before generating new code.
*   [ ] **Automate PRs:** Ensure Antigravity uses the `gh` CLI to open Pull Requests for every iteration.
*   [ ] **Telemetry Pipeline:** Ensure the Supercloud results are retrieved by Antigravity and pushed to the GCP Database.

## Phase 5: Execution & Evaluation (Comparative Run)
*   [ ] **The Dual 3-Hour Marathons:** Launch the recursive self-improvement loop for both Trial A (filtered) and Trial B (unfiltered) to compare optimization trajectory.
*   [ ] **Post-Mortem & Comparative Analysis:** Generate a report comparing the learning speed, trading performance, and cost-efficiency (AROI) of the self-improved agents under both data regimes.
