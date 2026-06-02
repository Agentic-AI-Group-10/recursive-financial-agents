import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    # Using pandas if available for a robust, standard implementation
    try:
        import pandas as pd
        data_arr = np.array(data, dtype=float)
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback to a numpy implementation if pandas is not available
        data_arr = np.array(data, dtype=float)
        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        ema_values[0] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(1, len(ema_values)):
            ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta >= 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    # Align series before subtraction
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    # Align series before subtraction
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=10):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy incorporating lessons on crash detection and capital preservation.
    1. Crisis Aversion: Enhanced with a velocity trigger (ROC) to detect crashes faster.
    2. Normal Trend-Following: Hardened with a dynamic stop-loss proxy to prevent "slow bleed" drawdowns.
    3. Sentiment: Refined keyword weights for more nuanced filtering.
    """
    # --- 1. Sentiment Analysis (Refined) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-impact positive
        "fed pivot": 3.5, "capitulation": 3.0, "quantitative easing": 3.0, "soft landing": 3.0,
        "rate cut": 2.5, "cooling inflation": 2.5, "stimulus": 2.5, "ai boom": 2.5,
        "panic selling": 2.5, "dovish": 2.0, "record high": 2.0, "strong earnings": 2.0,
        "recovery": 1.5, "beat estimates": 1.5, "upgrade": 1.5, "disinflation": 2.0,
        # High-impact negative
        "black swan": -4.0, "systemic risk": -4.0, "yield curve inversion": -4.0,
        "contagion": -3.5, "crisis": -3.5, "recession": -3.0, "stagflation": -3.0,
        "hot inflation": -3.0, "war": -3.0, "mania": -3.0, "hard landing": -3.0,
        "rate hike": -2.5, "quantitative tightening": -2.5, "vix spike": -2.5,
        "bankruptcy": -2.5, "geopolitical risk": -2.5, "bubble": -2.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "uncertainty": -1.5, "strong jobs report": -1.0, # Muted negative
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_PERIOD = 10
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, ATR_LONG + 1, ROC_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_10 = calculate_roc(all_prices, ROC_PERIOD)
    recent_peak = max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else current_price

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_10]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection (Enhanced) ---
    # Crisis is defined by a long-term downtrend combined with either a volatility spike OR a velocity spike.
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_rapid_decline = roc_10 < -8.0
    is_crisis_regime = is_long_term_downtrend and (is_high_volatility or is_rapid_decline)

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION
    if is_crisis_regime:
        # In a crisis, the primary goal is capital preservation. Exit all positions.
        # Do not attempt to buy dips.
        return "SELL"

    # REGIME 2: NORMAL / RECOVERY (Unified Trend-Following)
    is_primary_uptrend = current_price > sma_50

    # --- SELL LOGIC (Primary Exit & Capital Preservation) ---
    # This section is prioritized to manage risk before considering new buy signals.

    # A) Dynamic Stop-Loss Proxy: Exit if price drops significantly from a recent peak.
    # This addresses the "slow bleed" failure mode from past lessons.
    is_significant_drawdown = current_price < (recent_peak * 0.92) # 8% drop from 20-day high
    if is_significant_drawdown:
        return "SELL"

    # B) Trend Breakdown Exit: Sell when the primary trend breaks down with momentum confirmation.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.5
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # C) Profit-Taking Exit: Sell on extreme overbought conditions with fading momentum.
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Entry & Re-entry) ---
    # Buy only if no sell signals are present and conditions are favorable.
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0 # Slightly stricter threshold

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"