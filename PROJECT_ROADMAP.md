# PROJECT_ROADMAP.md

## Phase 1: Repository & Local Environment Setup
*   [ ] **Initialize Local Repo:** Clone `https://github.com/Agentic-AI-Group-10/recursive-financial-agents` to your local machine (Antigravity).
*   [ ] **Inject Context:** Save the `PROJECT_MANIFEST.md` and `RESEARCH_LINKS.md` files into the root directory.
*   [ ] **Setup Antigravity Workspace:** Open the folder in the Google Antigravity AI coding platform and ensure the agent can read the manifest.
*   [ ] **Configure GitHub CLI:** Ensure your local environment has the `gh` CLI authenticated so the agent can automatically open Pull Requests.

## Phase 2: Supercloud Preparation (The Foundry)
*   [ ] **Verify SSH Access:** Confirm you can SSH into the MIT Supercloud from your local terminal using your API keys/certificates.
*   [ ] **Stage Historical Data:** 
    *   Download 1-year of $SPY historical price data (1-minute or 5-minute intervals).
    *   Upload to `/home/gridsan/YOUR_USER/data/SPY_price.csv` on the Supercloud.
*   [ ] **Verify Local LLM:** Check which models are available on the cluster (e.g., Llama-3-8B). Note the path to the weights or the command to load the module.
*   [ ] **Create Conda Environment:** Create a `recursive_agents` environment on Supercloud with `pandas`, `numpy`, and `opentelemetry-sdk`.

## Phase 3: Infrastructure Development (The Engine)
*   [ ] **Build Backtest Engine:** Create `backtest_engine.py`. This script should:
    *   Load the $SPY CSV.
    *   Import `decide` from `strategy_logic.py`.
    *   Loop through data and calculate simulated profit.
    *   Output results to `results.json`.
*   [ ] **Draft Strategy Template:** Create a "Hello World" version of `strategy_logic.py` with a simple moving average crossover.
*   [ ] **Write Slurm Script:** Create `submit_job.sh` to:
    *   Execute the python engine.
    *   Capture GPU/CPU power draw (Telemetry).
    *   Save execution time.

## Phase 4: Antigravity Orchestration (The Bridge)
*   [ ] **Define the "Bridge" Skill:** In Antigravity, create a tool/script that:
    *   Copies `strategy_logic.py` to Supercloud via `scp`.
    *   Submits the job via `ssh`.
    *   Wait for completion and pulls `results.json` back to local `/telemetry`.
*   [ ] **Implement Telemetry Parser:** Write a local script to read `results.json` and calculate the **Efficiency Score** (Profit / Cost).
*   [ ] **Establish Long-Term Memory (LTM):** Setup a local JSON or vector store where Antigravity logs the "Alpha" (reasoning) and "Outcome" of every run.

## Phase 5: The Recursive Loop (Automation)
*   [ ] **Draft the "Architect" Prompt:** Create the system prompt for Gemini 1.5 Pro that instructs it to analyze `results.json` and the LTM to rewrite `strategy_logic.py`.
*   [ ] **Automate GitHub PRs:** Configure the orchestrator to:
    *   Create a new branch for every iteration.
    *   Commit the code changes.
    *   Open a Pull Request with the efficiency metrics in the description.
*   [ ] **The "Dry Run":** Execute one full cycle (Architect -> Bridge -> Supercloud -> Telemetry -> PR) and manually approve the PR.

## Phase 6: Scaling & Evaluation
*   [ ] **Set the "Run Clock":** Initiate the 6-hour/100-iteration marathon.
*   [ ] **Monitor Resource Budget:** Ensure the $50 "API/Energy" budget is tracked accurately as the iterations progress.
*   [ ] **Analyze Results:** Use the final telemetry history to create visualizations of "Efficiency over Time" for your final class presentation.