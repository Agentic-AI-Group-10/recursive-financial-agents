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

## Phase 2.5: Data Engineering (The "Simulation Pack")
*   [ ] **Acquire Datasets:** Download $SPY price data and Kaggle "Daily Financial News" locally.
*   [ ] **Build the Joiner:** Write a script in Antigravity to merge Price and News into a single `simulation_pack.csv`.
*   [ ] **Temporal Validation:** Verify that news timestamps are correctly aligned to prevent the agent from seeing "future" headlines during the backtest.
*   [ ] **Deploy to Foundry:** Upload the `simulation_pack.csv` to the MIT Supercloud `/data` directory.

## Phase 3: Supercloud Preparation (The Foundry)
*   [ ] **Stage Historical Data:** Upload $SPY price data to the Supercloud `/data` directory. #do not know if that's actually a data source we are using?
*   [ ] **Verify Local LLM:** Identify the path for Llama-3/Mistral on the Supercloud.
*   [ ] **Finalize Engine:** Upload `backtest_engine.py` and `submit_job.sh` to Supercloud.

## Phase 4: Recursive Loop Development
*   [ ] **Build the "Architect" Skill:** In Antigravity, write the logic that queries GCP LTM for past failures before generating new code.
*   [ ] **Automate PRs:** Ensure Antigravity uses the `gh` CLI to open Pull Requests for every iteration.
*   [ ] **Telemetry Pipeline:** Ensure the Supercloud results are retrieved by Antigravity and pushed to the GCP Database.

## Phase 5: Execution & Evaluation
*   [ ] **The 6-Hour Marathon:** Launch the recursive loop and monitor the GCP Dashboard.
*   [ ] **Post-Mortem:** Use the Telemetry Hub data to generate the final efficiency report.
