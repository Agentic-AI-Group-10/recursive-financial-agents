# PROJECT_MANIFEST.md

## 1. Project Purpose & Vision
The goal is to design and deploy a **Recursive Agentic System** that performs autonomous Research and Development (R&D) to optimize a business plan. Due to network and safety constraints, the "Business" is simulated as **Quantitative Trading ($SPY)**. 

The system aims to achieve **Recursive Self-Improvement (RSI)**: an orchestrator agent generates strategy logic, executes it in a sandbox, collects telemetry (profit, energy, tokens), and uses those "lessons learned" to rewrite the next version of the strategy to be more efficient.

## 2. Technical Architecture (The Quad-Node Model)
The project operates across three integrated nodes (optimized for local GCP VM execution & Cloud APIs):
*   **The Architect (Google AI Studio):** Uses Gemini 1.5 Pro/Flash for high-level strategy and code generation.
*   **The Orchestrator (Antigravity):** The local development agent. Manages the GitHub repository, branch generation, and code deployment on the VM.
*   **The Nerve Center & Backtest Sandbox (Google Cloud Platform VM):** A persistent VM instance funded by $300 in GCP credits. 
    *   Hosts the centralized **PostgreSQL/pgvector** database for Long-Term Memory (LTM).
    *   Hosts the **Backtesting Engine (`backtest_engine.py`)** for fast sandbox evaluation (2-second CPU runs).
    *   Hosts the **Telemetry Dashboard** (Streamlit app) to track and visualize Efficiency Scores.
*   **The Foundry (Cloud API Endpoints):** Replaces the heavy, queue-limited local compute (MIT Supercloud) with high-speed, cost-efficient APIs:
    *   **Google AI Studio (Gemini 1.5 Flash):** Primary sentiment engine (Fast, large context, completely free tier).
    *   **DeepInfra API Aggregator:** Backup engine for running open-weights models (like Llama-3-8B) on demand at sub-penny per-token pricing ($0.055 per 1M tokens).

## 3. Core Research Pillars
*   **Agentic ROI (AROI) & C2R:** We prioritize **Compute-to-Revenue Efficiency**. Profit is measured against the cost of the "thinking" (Tokens) and the "doing" (Energy/Joules).
*   **2026 Agentic Economy:** We simulate future protocols including **AP2/x402** and **Judge-Agent Architecture** (where Gemini 1.5 Pro must audit code for safety/legality).
*   **Energy Metrics:** We estimate CPU/GPU energy consumption based on active thread execution and local GCP machine TDP profiles to calculate the carbon and financial cost of compute.

## 4. Key Design Decisions
*   **Domain:** Quantitative Trading of a single ticker ($SPY).
*   **Strategy Interface:** The "Worker Agent" only writes the `decide()` function within a fixed `strategy_logic.py` file.
*   **The Meta-Loop:** The system will run for **100 generations or 6 hours**, whichever comes first.
*   **Multi-Regime Validation & Dual-Dataset Comparative Trial:** To guarantee strategy robustness, our experiment operates across two distinct market regimes and two data densities:
    *   **In-Sample Training Regime (2017–2018 "Normal News Cycle"):** We run the recursive optimization loop under stable market conditions. We compare learning curves and agentic efficiency on a high-signal filtered set (`simulation_pack_filtered_normal.csv`) vs. a full unfiltered set (`simulation_pack_unfiltered_normal.csv` with a 100-headline safety cap).
    *   **Out-of-Sample Stress-Testing Regime (2019–2020 "Covid Black Swan"):** We take the best-evolved agents and evaluate them on highly volatile crash data (`simulation_pack_filtered.csv` and `simulation_pack_unfiltered.csv`) without further training to measure real-world generalized trading robustness.
*   **Safety & Review:** Antigravity is instructed to **automatically open Pull Requests (PRs)**. Humans must review and approve these PRs.
*   **Budgeting:**
    *   **$50 Startup Fund:** Allocated for paid API tokens (DeepInfra/Google AI Studio) and energy estimation costs.
    *   **$300 GCP Credits:** Allocated for persistent cloud infrastructure (GCP Compute Engine, PostgreSQL, and Streamlit dashboards).

## 5. Measurement & Success Metrics
The system optimizes for the **Efficiency Score** (Net AROI):
$$Score = \frac{Simulated \ Profit \ (USD)}{(Tokens \times Price) + (Joules \times Carbon \ Cost)}$$

## 6. Directory Structure Reference
*   `/engine`: Permanent backtesting infrastructure executing locally on GCP VM.
*   `/strategies`: The `strategy_logic.py` file evolved by the AI.
*   `/telemetry`: `results.json` and performance curve charts.
*   `/database`: PostgreSQL `schema.sql` (hosted on GCP VM).
*   `/docs`: PR summaries and conversation logs.
*   `/data`: Contains local datasets, including `spy_prices.csv`, `simulation_pack_filtered.csv`, and `simulation_pack_unfiltered.csv` (excluded from git).

## 7. Research Links & Resources (Summary)
- GitHub: https://github.com/Agentic-AI-Group-10/recursive-financial-agents
- DeepInfra Docs: https://deepinfra.com/docs
- Google AI Studio: https://aistudio.google.com/
- Telemetry Tools: Streamlit Dashboard | PostgreSQL LTM

## 8. Instructions for Antigravity Agent
1.  **Orchestration:** Run the backtest engine locally on the GCP VM.
2.  **Versioning:** Every iteration must be on a new branch. Document the "Alpha" (reasoning) for every change in the PR body.
3.  **Data Integrity:** Ensure the local PostgreSQL LTM is updated immediately after a backtest run finishes.
4.  **Optimization:** If the Efficiency Score drops, consult the GCP PostgreSQL LTM to identify previous similar failures.