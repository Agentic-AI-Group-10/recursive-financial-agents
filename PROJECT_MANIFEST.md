# PROJECT_MANIFEST.md

## 1. Project Purpose & Vision
The goal is to design and deploy a **Recursive Agentic System** that performs autonomous Research and Development (R&D) to optimize a business plan. Due to network and safety constraints, the "Business" is simulated as **Quantitative Trading ($SPY)**. 

The system aims to achieve **Recursive Self-Improvement (RSI)**: an orchestrator agent generates strategy logic, executes it in a sandbox, collects telemetry (profit, energy, tokens), and uses those "lessons learned" to rewrite the next version of the strategy to be more efficient.

## 2. Technical Architecture (The Quad-Node Model)
The project operates across four distinct environments:
*   **The Architect (Google AI Studio):** Uses Gemini 1.5 Pro/Flash for high-level strategy and code generation.
*   **The Orchestrator (Antigravity):** The local development agent. Manages the GitHub repository and acts as the SSH bridge between all nodes.
*   **The Nerve Center (Google Cloud Platform - NEW):** A persistent VM instance funded by $300 in GCP credits. 
    *   Hosts a centralized **PostgreSQL/pgvector** database for Long-Term Memory (LTM).
    *   Hosts a **Telemetry Dashboard** (e.g., SigNoz or a custom Streamlit app) to track Efficiency Scores.
*   **The Foundry (MIT Supercloud):** An air-gapped compute cluster where the simulation runs. It hosts historical data, the backtesting engine, and local LLMs (e.g., Llama-3).

## 3. Core Research Pillars
*   **Agentic ROI (AROI) & C2R:** We prioritize **Compute-to-Revenue Efficiency**. Profit is measured against the cost of the "thinking" (Tokens) and the "doing" (Energy/Joules).
*   **2026 Agentic Economy:** We simulate future protocols including **AP2/x402** and **Judge-Agent Architecture** (where Gemini 1.5 Pro must audit code for safety/legality).
*   **Energy Metrics:** Using MIT research, we estimate energy consumption based on GPU/CPU wattage and execution time to calculate the carbon and financial cost of compute.

## 4. Key Design Decisions
*   **Domain:** Quantitative Trading of a single ticker ($SPY).
*   **Strategy Interface:** The "Worker Agent" only writes the `decide()` function within a fixed `strategy_logic.py` file.
*   **The Meta-Loop:** The system will run for **100 generations or 6 hours**, whichever comes first.
*   **Safety & Review:** Antigravity is instructed to **automatically open Pull Requests (PRs)**. Humans must review and approve these PRs.
*   **Budgeting:**
    *   **$50 Startup Fund:** Allocated for Gemini AI API tokens and energy estimation costs.
    *   **$300 GCP Credits:** Allocated for persistent cloud infrastructure (DBs, Dashboards, and Proxy Servers).

## 5. Measurement & Success Metrics
The system optimizes for the **Efficiency Score**:
$$Score = \frac{Simulated \ Profit \ (USD)}{(Tokens \times Price) + (Joules \times Carbon \ Cost)}$$

## 6. Directory Structure Reference
*   `/engine`: Permanent backtesting infrastructure (MIT Supercloud).
*   `/strategies`: The `strategy_logic.py` file evolved by the AI.
*   `/telemetry`: `results.json` and OTel logs.
*   `/ltm`: Long-Term Memory (hosted on **GCP PostgreSQL**).
*   `/docs`: PR summaries and conversation logs.

## 7. Research Links & Resources (Summary)
- GitHub: https://github.com/Agentic-AI-Group-10/recursive-financial-agents
- MIT Supercloud Docs: https://mit-supercloud.github.io/supercloud-docs/
- Energy/Policy Research: ArXiv 2401.09646 | MIT AIA | MIT DCC
- Telemetry Tools: SigNoz (OTel) | Galileo AI

## 8. Instructions for Antigravity Agent
1.  **Orchestration:** Use the `supercloud_bridge` skill to manage `scp` and `ssh` commands. 
2.  **Versioning:** Every iteration must be on a new branch. Document the "Alpha" (reasoning) for every change in the PR body.
3.  **Data Integrity:** Ensure the GCP-hosted LTM is updated immediately after a Supercloud job finishes.
4.  **Optimization:** If the Efficiency Score drops, consult the GCP LTM to identify previous similar failures.