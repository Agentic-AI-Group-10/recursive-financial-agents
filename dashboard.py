#!/usr/bin/env python3
# ==============================================================================
# Streamlit Performance & Telemetry Dashboard
# Location: /home/ow9800/recursive-financial-agents/dashboard.py
# ==============================================================================
# This dashboard connects to PostgreSQL to query live agent evolution progress,
# compare strategy versions, track compute and token metrics, and plot equity curves.

import os
import json
import psycopg2
import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError:
    print("Streamlit not found. Please run: pip install streamlit")
    # Stub streamlit so the code compiles and can be previewed/edited safely
    class StreamlitStub:
        def __getattr__(self, name):
            def stub(*args, **kwargs):
                return None
            return stub
    st = StreamlitStub()

# Page configuration for rich professional aesthetics
st.set_page_config(
    page_title="Recursive Agentic Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS styling for premium look and feel
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main {
        background: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DB Queries
# ==============================================================================
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname="recursive_trading",
            host="/var/run/postgresql",
            user="ow9800"
        )
    except Exception as e:
        st.error(f"PostgreSQL connection failed: {e}")
        return None


def load_all_runs():
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
        
    query = """
        SELECT 
            r.run_id,
            s.generation,
            r.regime,
            s.branch_name,
            r.total_return_pct,
            r.benchmark_return_pct,
            r.alpha_vs_benchmark,
            r.max_drawdown_pct,
            r.sharpe_ratio_annualized,
            r.win_rate_pct,
            r.total_trades_executed,
            r.total_closed_trades,
            r.elapsed_seconds,
            r.total_energy_joules,
            r.token_cost_usd,
            r.energy_cost_usd,
            r.total_compute_cost_usd,
            r.net_profit_loss_usd,
            r.efficiency_score_aroi,
            s.rationale,
            r.trade_log_json,
            s.code_content,
            s.strategy_id
        FROM runs r
        JOIN strategies s ON r.strategy_id = s.strategy_id
        ORDER BY s.generation DESC, r.regime ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_lessons():
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
        
    query = """
        SELECT 
            l.lesson_id,
            s.generation,
            l.summary,
            l.sentiment_pattern,
            l.key_failure_cause,
            l.created_at
        FROM lessons_learned l
        JOIN strategies s ON l.strategy_id = s.strategy_id
        ORDER BY s.generation DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ==============================================================================
# Main Dashboard UI
# ==============================================================================
def draw_ui():
    st.title("🌀 Recursive Quant-Agent Evolution Center")
    st.markdown("### S&P 500 Strategy Self-Improvement Loop Real-Time Monitoring Telemetry")
    st.markdown("---")

    # Load Data
    runs_df = load_all_runs()
    lessons_df = load_lessons()

    if runs_df.empty:
        st.warning("No backtest runs found in the database. Run the orchestrator loop first!")
        st.info("To start the loop, run: `python3 engine/orchestrator.py --generations 5`")
        return

    # Total cumulative statistics
    total_compute_cost = runs_df['total_compute_cost_usd'].sum()
    total_runs = len(runs_df)
    max_aroi = runs_df['efficiency_score_aroi'].max()
    best_return = runs_df['total_return_pct'].max()
    best_strat_row = runs_df[runs_df['total_return_pct'] == best_return].iloc[0] if not runs_df.empty else None

    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Evolutionary Runs", value=f"{total_runs}")
    with col2:
        st.metric(label="Best Strategic Return", value=f"{best_return:.2f}%", 
                  delta=f"Alpha: {best_strat_row['alpha_vs_benchmark']:.2f}%" if best_strat_row is not None else None)
    with col3:
        st.metric(label="Peak Agentic ROI (AROI)", value=f"{max_aroi:.2f}x")
    with col4:
        st.metric(label="Cumulative Compute Cost", value=f"${total_compute_cost:.4f} USD")

    st.markdown("---")

    # Layout: Live Runs Table & Evolutionary Rationale
    t1, t2, t3 = st.tabs(["📊 Performance Leaderboard", "🧠 Semantic Long-Term Memory (LTM)", "💻 Strategy Code Inspection"])

    with t1:
        st.subheader("Leaderboard: Sorted by Return & Agentic Efficiency")
        
        # Selection of regime to filter
        regimes = runs_df['regime'].unique().tolist()
        selected_regime = st.selectbox("Select Market Regime Profile", options=regimes)
        filtered_df = runs_df[runs_df['regime'] == selected_regime].sort_values(by='total_return_pct', ascending=False)
        
        # Display simplified table
        display_cols = [
            'generation', 'branch_name', 'total_return_pct', 'benchmark_return_pct', 
            'alpha_vs_benchmark', 'max_drawdown_pct', 'sharpe_ratio_annualized', 
            'win_rate_pct', 'total_trades_executed', 'total_compute_cost_usd', 'efficiency_score_aroi'
        ]
        
        formatted_df = filtered_df[display_cols].copy()
        formatted_df.rename(columns={
            'generation': 'Gen',
            'branch_name': 'Git Branch',
            'total_return_pct': 'Return (%)',
            'benchmark_return_pct': 'Benchmark (%)',
            'alpha_vs_benchmark': 'Alpha (%)',
            'max_drawdown_pct': 'Max DD (%)',
            'sharpe_ratio_annualized': 'Sharpe Ratio',
            'win_rate_pct': 'Win Rate (%)',
            'total_trades_executed': 'Trades',
            'total_compute_cost_usd': 'Cost ($)',
            'efficiency_score_aroi': 'AROI Score'
        }, inplace=True)
        
        st.dataframe(formatted_df.style.format({
            'Return (%)': '{:.2f}%',
            'Benchmark (%)': '{:.2f}%',
            'Alpha (%)': '{:.2f}%',
            'Max DD (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.3f}',
            'Win Rate (%)': '{:.1f}%',
            'Cost ($)': '${:.6f}',
            'AROI Score': '{:.2f}x'
        }), use_container_width=True)

    with t2:
        st.subheader("Lessons Learned and Post-Mortems")
        if not lessons_df.empty:
            for idx, r in lessons_df.iterrows():
                with st.expander(f"🧠 Generation {r['generation']} Lesson Learned Summary"):
                    st.markdown(f"**Qualitative Post-Mortem:**\n{r['summary']}")
                    st.markdown(f"**Observed Sentiment Pattern Triggers:**\n`{r['sentiment_pattern']}`")
                    if r['key_failure_cause']:
                        st.markdown(f"**Failure Cause / Drawdown Diagnostic:**\n`{r['key_failure_cause']}`")
        else:
            st.info("No lessons logged yet in the `lessons_learned` table.")

    with t3:
        st.subheader("Strategy Source Code Comparison")
        
        # Selecting generations to inspect
        generations_list = sorted(runs_df['generation'].unique().tolist())
        selected_gen = st.selectbox("Select Generation to Inspect Source Code", options=generations_list)
        
        gen_data = runs_df[runs_df['generation'] == selected_gen].iloc[0]
        
        st.markdown(f"### Generation {selected_gen} (`{gen_data['branch_name']}`)")
        st.markdown(f"**Architect Rationale:**\n*{gen_data['rationale']}*")
        
        # Display Code Block
        st.code(gen_data['code_content'], language='python')
        
        # Detailed Trade Logs for Selected Gen
        st.markdown("### Generation Detailed Trade Log")
        try:
            trade_logs = json.loads(gen_data['trade_log_json']) if isinstance(gen_data['trade_log_json'], str) else gen_data['trade_log_json']
            if trade_logs:
                trade_df = pd.DataFrame(trade_logs)
                st.dataframe(trade_df, use_container_width=True)
            else:
                st.info("No trades executed by this generation.")
        except Exception as e:
            st.error(f"Failed to display trade log: {e}")

    # Sidebar parameters and live image plotting
    st.sidebar.title("Configuration & Controls")
    st.sidebar.info("Dashboard auto-updates from PostgreSQL LTM.")
    
    # Show active parent strategy info
    st.sidebar.subheader("Active Evolution Base")
    best_gen_overall = runs_df.sort_values(by='efficiency_score_aroi', ascending=False).iloc[0]
    st.sidebar.markdown(f"**Best AROI Generation:** Gen {best_gen_overall['generation']}")
    st.sidebar.markdown(f"**Best AROI:** {best_gen_overall['efficiency_score_aroi']:.2f}x")
    st.sidebar.markdown(f"**Best Return:** {best_gen_overall['total_return_pct']:.2f}%")

    # Display dynamic chart from local file
    st.sidebar.subheader("Selected Run Plot")
    chart_gen = st.sidebar.selectbox("Plot Strategy Equity Curve", options=generations_list, index=len(generations_list)-1)
    chart_regime = st.sidebar.selectbox("Plot Regime", options=["Normal", "Stress"])
    
    regime_suffix = "normal" if chart_regime == "Normal" else "stress"
    expected_image_path = f"telemetry/gen_{chart_gen}_{regime_suffix}_equity.png"
    
    if os.path.exists(expected_image_path):
        st.sidebar.image(expected_image_path, caption=f"Gen {chart_gen} {chart_regime} Equity Curve", use_column_width=True)
    else:
        st.sidebar.info(f"No plot file found for Gen {chart_gen} {chart_regime}. If the orchestrator is currently running, plots will generate sequentially.")


# Stub execution wrapper
if __name__ == "__main__":
    if 'StreamlitStub' not in globals() and 'StreamlitStub' not in locals():
        draw_ui()
