# recursive-financial-agents
AI Agent class project repository for Group 10: Non-Monetary Performance Metrics for Recursive Financial Agents.

---

# About this project

This repository contains the complete implementation, database schemas, data pipelines, and telemetry dashboards for Group 10's study on **Non-Monetary Performance Metrics for Recursive Financial Agents**. The platform tests how autonomous Large Language Models (LLMs) can cooperate as self-correcting algorithmic "Architects" to recursively evolve production-grade trading strategies under strict risk and environmental footprints.

## 🎯 Goal
The objective is to establish an autonomous, closed-loop evolutionary system where an LLM agent recursively drafts, tests, and refines quantitative trading indicators. Instead of human engineering, the model reads historical trade failures and semantic lessons from a PostgreSQL Long-Term Memory (LTM) database and self-corrects code to maximize risk-adjusted performance across varying market conditions.

## 🏗️ Architecture
The system is constructed with a decoupled, high-resilience architecture:
*   **PostgreSQL LTM (Nerve Center)**: Stores strategies, backtest metrics, and semantic summaries of "lessons learned" across generations to prevent the LLM from repeating past code errors. *(Schema initialized in [database/schema.sql](file:///home/ow9800/recursive-financial-agents/database/schema.sql))*
*   **Backtesting Engine (`backtest_engine.py`)**: Evaluates generated Python strategies on SPY stock data combined with Kaggle Daily Financial News under tight, look-ahead guarded point-in-time left-join compliance.
*   **Multi-Model Orchestrator (`engine/orchestrator.py`)**: Runs the recursive loop, isolating and storing database metrics for each individual LLM trial as standalone PostgreSQL SQL backups.

## ⚡ Non-Monetary Metrics: AROI
While traditional quant trading evaluates pure financial alpha (Returns, Sharpe, Drawdown), this project pioneers **Agentic ROI (AROI)**, a multi-dimensional metric evaluating computational and environmental overhead:
$$AROI = \frac{\text{Net Trading Profit (USD)}}{\text{Hardware Energy Cost (USD)} + \text{LLM API Cost (USD)}}$$
The backtest engine profiles physical hardware (CPU/Memory power consumption in Joules), maps it to environmental carbon offsetting costs (USD), and aggregates LLM token API expenses. This penalizes computationally heavy models and rewards clean, green, and highly optimized trading agents.

## 🏆 Key Conclusions
*   **Grand Champion**: **Gemini 2.5 Pro (Filtered)** successfully evolved a sophisticated momentum strategy incorporating RSI, Rate of Change (ROC) velocity filters, and adaptive sentiment thresholds. Under the normal market regime, it achieved **22.23% return** (a 220% improvement over the 6.16% baseline), a **1.63 Sharpe**, and cut drawdown in half to **-6.87%**.
*   **Stress Adaptability**: Tested against the Covid-19 Crash (Stress Regime), the champion strategy demonstrated true regime-invariant risk mitigation, generating **23.90% return** while keeping drawdown at **-8.86%**.
*   **Energy Efficiency Champion**: While Pro delivered peak financial returns, **Gemini 2.5 Flash** achieved the highest overall **AROI**, demonstrating that smaller, lightweight models are significantly more cost-efficient per unit of trading return.

---

# Updates

**06/04/2026 09:00** - Concluded the 12 multi-model evolutionary marathons. Saved isolated PostgreSQL backups under `/database`. Extracted and deployed the **Gemini 2.5 Pro Generation 78 Grand Champion Strategy** directly into the workspace active code (`strategies/strategy_logic.py`). Generated static high-fidelity Matplotlib charts and created an interactive HTML/JS web dashboard under `/visualizations` to examine Return, AROI, and hardware power (Joules) footprint curves. Compiled extensive project writeups and post-mortem reports.

**05/26/2026 18:23** - I think I added you to the Google project? And possibly another Google cloud project too.

**05/26/2026 18:10** - Added ai_notes for things that don't make it in the markdown outputs in my AI chats (more than you would think).

**05/26/2026 15:51** - Uploaded new project manifest, road map, cloud guide, prompt suggestion for AI prompting.

**05/26/2026 17:48** - Okay I fixed the billing, but also messed it up. If you get a message saying the account ran out of money, let me know and I will direct it to a different pool of money. Also I accidentally got $300 of Google Cloud credits as a trial for an account I didn't realize I was opening. So I guess I will make a project there too.

**05/26/2026 17:31** - Added some structure to the README. I am also using this as a way to model how to summarize the changes in each iteration of the project for when the coding agent is reviewing our work and building on it.

**05/26/2026 17:28** - Uploaded research URL document. Wasn't sure if we needed this, still not sure, but it's available!

**05/26/2026 17:25** - Uploaded project manifest and roadmap. Pending some billing tweaks, we should be good to get coding started.
