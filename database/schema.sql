-- ==============================================================================
-- PostgreSQL Database Schema with pgvector for Long-Term Memory (LTM)
-- Location: database/schema.sql
-- ==============================================================================
-- This schema initializes the "Nerve Center" GCP database. It stores historical
-- strategy performance, compute costs, trade logs, and high-dimensional 
-- lessons-learned embeddings for recursive self-improvement retrieval.

-- 1. Enable pgvector extension (requires pgvector installation on Ubuntu VM)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 2. Create Strategies Table
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation INTEGER NOT NULL,
    branch_name VARCHAR(255) NOT NULL UNIQUE,
    code_content TEXT NOT NULL,
    code_hash VARCHAR(64) NOT NULL,
    rationale TEXT NOT NULL,
    parent_strategy_id UUID REFERENCES strategies(strategy_id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for parent strategy hierarchy traversal
CREATE INDEX IF NOT EXISTS idx_strategies_parent_id ON strategies(parent_strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategies_generation ON strategies(generation);

-- 3. Create Runs (Backtesting Results) Table
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    regime VARCHAR(100) NOT NULL,                    -- 'Normal_Filtered', 'Normal_Unfiltered', 'Covid_Filtered', 'Covid_Unfiltered'
    initial_portfolio_value NUMERIC(15, 2) NOT NULL,
    final_portfolio_value NUMERIC(15, 2) NOT NULL,
    total_return_pct NUMERIC(10, 4) NOT NULL,
    benchmark_return_pct NUMERIC(10, 4) NOT NULL,
    alpha_vs_benchmark NUMERIC(10, 4) NOT NULL,
    max_drawdown_pct NUMERIC(10, 4) NOT NULL,
    sharpe_ratio_annualized NUMERIC(10, 4) NOT NULL,
    win_rate_pct NUMERIC(10, 2) NOT NULL,
    total_trades_executed INTEGER NOT NULL,
    total_closed_trades INTEGER NOT NULL,
    
    -- Telemetry & Compute Costs
    elapsed_seconds NUMERIC(12, 4) NOT NULL,
    total_energy_joules NUMERIC(15, 2) NOT NULL,
    measured_via_hardware BOOLEAN NOT NULL DEFAULT FALSE,
    input_tokens_consumed INTEGER NOT NULL DEFAULT 0,
    output_tokens_consumed INTEGER NOT NULL DEFAULT 0,
    total_llm_calls INTEGER NOT NULL DEFAULT 0,
    
    -- Calculated Costs (USD)
    token_cost_usd NUMERIC(15, 6) NOT NULL DEFAULT 0.000000,
    energy_cost_usd NUMERIC(15, 6) NOT NULL DEFAULT 0.000000,
    total_compute_cost_usd NUMERIC(15, 6) NOT NULL DEFAULT 0.000000,
    
    -- Scoring
    net_profit_loss_usd NUMERIC(15, 2) NOT NULL,
    efficiency_score_aroi NUMERIC(18, 4) NOT NULL,  -- Net Profit / Compute Cost
    
    -- Complete Trade Log & Raw Telemetry
    trade_log_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance querying and regime comparisons
CREATE INDEX IF NOT EXISTS idx_runs_strategy_id ON runs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_runs_regime ON runs(regime);
CREATE INDEX IF NOT EXISTS idx_runs_aroi ON runs(efficiency_score_aroi);
CREATE INDEX IF NOT EXISTS idx_runs_total_return ON runs(total_return_pct);

-- 4. Create Lessons Learned (Semantic LTM) Table
CREATE TABLE IF NOT EXISTS lessons_learned (
    lesson_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    
    -- Textual descriptions for LLM context injection
    summary TEXT NOT NULL,                           -- Overall qualitative performance analysis
    sentiment_pattern TEXT NOT NULL,                 -- Sentiment triggers observed in news context
    key_failure_cause TEXT,                          -- Diagnostic details on drawdowns or false-signals
    
    -- Vector Embedding (768 dimensions for Gemini text-embedding-004)
    -- Change dimension size to 1536 if using OpenAI text-embedding-3-small/ada-002
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast cosine similarity search (using IVFFlat index or HNSW for pgvector)
-- Note: A cosine distance index is created. Adjust lists parameter according to data scaling.
CREATE INDEX IF NOT EXISTS idx_lessons_embedding_cosine ON lessons_learned USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_lessons_strategy_id ON lessons_learned(strategy_id);
CREATE INDEX IF NOT EXISTS idx_lessons_run_id ON lessons_learned(run_id);
