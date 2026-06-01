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

## Phase 2.5: Data Engineering (Dual-Dataset Simulation Packs)
*   [x] **Acquire Datasets:** Download $SPY price data and Kaggle "Daily Financial News" locally.
*   [x] **Build the Joiner:** Write a temporal point-in-time left-joiner with look-ahead guards and timezone normalization (`build_simulation_pack.py`).
*   [x] **Dual Simulation Pack Generation:**
    *   [x] **High-Signal Filtered Pack:** Generate `simulation_pack_filtered.csv` (macro-economic keywords + S&P 500 mega-caps).
    *   [x] **Full Unfiltered Pack:** Generate `simulation_pack_unfiltered.csv` (all headlines, with a 100 headline-per-bar safety cap).
*   [ ] **Deploy to Foundry:** Upload both simulation packs to the MIT Supercloud `/data` directory.

## Phase 2.6: Experimental Design (Filtered vs. Unfiltered Trials)
*   [ ] **Comparative Test Matrix:** Define the trial configurations for both datasets.
*   [ ] **Trial A (High-Signal Benchmark):** Run the self-improving agents on the `simulation_pack_filtered.csv` dataset.
*   [ ] **Trial B (Full Context Benchmark):** Run the same agents on the `simulation_pack_unfiltered.csv` dataset.
*   [ ] **Comparative Performance Metrics:** Formulate the final analysis comparing:
    *   **Monetary Return:** Returns, Sharpe Ratio, Maximum Drawdown.
    *   **Context Efficiency:** Average tokens consumed per trading decision.
    *   **Agentic ROI (AROI):** Net return of the strategy minus AI token & energy costs.
    *   **Inference Latency:** Iteration speed (seconds per decision).

## Phase 3: Supercloud Preparation (The Foundry)
*   [ ] **Stage Historical Data:** Upload both simulation pack CSVs to Supercloud `/data`.
*   [ ] **Verify Local LLM:** Identify the path for Llama-3/Mistral on the Supercloud.
*   [ ] **Finalize Engine:** Upload `backtest_engine.py` and `submit_job.sh` to Supercloud.

## Phase 4: Recursive Loop Development
*   [ ] **Build the "Architect" Skill:** In Antigravity, write the logic that queries GCP LTM for past failures before generating new code.
*   [ ] **Automate PRs:** Ensure Antigravity uses the `gh` CLI to open Pull Requests for every iteration.
*   [ ] **Telemetry Pipeline:** Ensure the Supercloud results are retrieved by Antigravity and pushed to the GCP Database.

## Phase 5: Execution & Evaluation (Comparative Run)
*   [ ] **The Dual 3-Hour Marathons:** Launch the recursive self-improvement loop for both Trial A (filtered) and Trial B (unfiltered) to compare optimization trajectory.
*   [ ] **Post-Mortem & Comparative Analysis:** Generate a report comparing the learning speed, trading performance, and cost-efficiency (AROI) of the self-improved agents under both data regimes.
