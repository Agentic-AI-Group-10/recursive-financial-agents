# PROJECT_MANIFEST.md

## 1. Project Purpose & Vision
The goal is to design and deploy a **Recursive Agentic System** that performs autonomous Research and Development (R&D) to optimize a business plan. Due to network and safety constraints, the "Business" is simulated as **Quantitative Trading ($SPY)**. 

The system aims to achieve **Recursive Self-Improvement (RSI)**: an orchestrator agent generates strategy logic, executes it in a sandbox, collects telemetry (profit, energy, tokens), and uses those "lessons learned" to rewrite the next version of the strategy to be more efficient.

## 2. Technical Architecture (The "Bridge" Model)
The project operates across three distinct environments:
*   **The Orchestrator (Antigravity):** The local command center. It uses Gemini 1.5 Pro to "think," manages the GitHub repository, and acts as the SSH bridge to the Supercloud.
*   **The Foundry (MIT Supercloud):** An air-gapped compute cluster where the simulation runs. It hosts historical data, the backtesting engine, and local LLMs (e.g., Llama-3) for mid-run logic.
*   **The Memory (GitHub):** Stores all code, prompts, telemetry history, and "Lessons Learned" logs.

## 3. Core Research Pillars
*   **Agentic ROI (AROI) & C2R:** We prioritize **Compute-to-Revenue Efficiency**. Profit is measured against the cost of the "thinking" (Tokens) and the "doing" (Energy/Joules).
*   **2026 Agentic Economy:** We simulate future protocols:
    *   **AP2/x402:** Programmatic, HTTP-native payment/authorization for micro-services.
    *   **Judge-Agent Architecture:** A high-reasoning model (Gemini 1.5 Pro) must "sign off" on any generated code for safety and logic before it is pushed to the cluster.
*   **Energy Metrics:** Using MIT research (Sloan/DCC/AIA), we estimate energy consumption based on GPU/CPU wattage and execution time to calculate the carbon and financial cost of compute.

## 4. Key Design Decisions
*   **Domain:** Quantitative Trading of a single ticker ($SPY).
*   **Strategy Interface (Option B):** The "Worker Agent" does **not** write full scripts. It only writes the `decide()` function within a fixed `strategy_logic.py` file. This is then imported by a static `backtest_engine.py`.
*   **The Meta-Loop:** The system will run for **100 generations or 6 hours**, whichever comes first.
*   **Safety & Review:** Antigravity is instructed to **automatically open Pull Requests (PRs)** for every strategy improvement. Humans must review and approve these PRs, though the agent handles all documentation and rationale.
*   **Budget:** A $50 startup "investment" is allocated strictly as an **API and Compute Budget** (tokens and energy costs).

## 5. Measurement & Success Metrics
The system optimizes for the **Efficiency Score**:
$$Score = \frac{Simulated \ Profit \ (USD)}{(Tokens \times Price) + (Joules \times Carbon \ Cost)}$$

## 6. Directory Structure Reference
*   `/engine`: Permanent backtesting infrastructure (MIT Supercloud).
*   `/strategies`: The `strategy_logic.py` file evolved by the AI.
*   `/telemetry`: `results.json` and OTel logs capturing Time, Tokens, and Energy.
*   `/ltm`: Long-Term Memory vector store containing "Lessons Learned" from iterations.
*   `/docs`: PR summaries and conversation logs for human collaboration.

## 7. Research Links & Resources

### Infrastructure & Policy
- GitHub: https://github.com/Agentic-AI-Group-10/recursive-financial-agents
- Supercloud Docs: https://mit-supercloud.github.io/supercloud-docs/
- Coincub (2026 Economy): https://coincub.com/

### Energy & Sustainability
- MIT AIA: https://aia.mit.edu/research/
- MIT DCC: https://dcc.mit.edu/data/
- MIT Sloan: https://mitsloan.mit.edu/ideas-made-to-matter/ai-has-high-data-center-energy-costs-there-are-solutions
- ArXiv (Power): https://arxiv.org/abs/2401.09646
- ArXiv (Footprint): https://arxiv.org/pdf/2108.02037
- Podcast: https://podcasts.apple.com/us/podcast/the-latitude-stage-how-ai-changes-our-digital-energy/id1794164180?i=1000722815530
- Compute Partners: https://baycompute.com/ | https://www.radium.cloud/

### Telemetry & Eval
- SigNoz: https://signoz.io/
- NVIDIA NeMo: https://developer.nvidia.com/nemo-framework
- Galileo AI: https://www.galileo.ai/

## 8. Instructions for Antigravity Agent
1.  **Orchestration:** Use the `supercloud_bridge` skill to manage `scp` and `ssh` commands. 
2.  **Versioning:** Every iteration must be on a new branch. Document the "Alpha" (reasoning) for every change in the PR body.
3.  **Simulation:** Do not attempt to reach the internet from the Supercloud. All data ($SPY CSV) must be pre-staged in the `/data` directory on the cluster.
4.  **Optimization:** If the Efficiency Score drops, consult the `/ltm` folder to see if a similar failure has occurred in past iterations.