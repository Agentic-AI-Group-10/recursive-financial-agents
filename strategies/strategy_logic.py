import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        ema_values = np.zeros(len(data_arr), dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
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
    ema_short = calculate_ema_series(prices, short_period)
    ema_long = calculate_ema_series(prices, long_period)
    macd_line = ema_short[long_period-1:] - ema_long[long_period-1:]
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[signal_period-1:] - signal_line[signal_period-1:]
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    prices_arr = np.array(prices[-period:], dtype=float)
    sma = np.mean(prices_arr)
    std_dev = np.std(prices_arr)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a multi-regime model to improve adaptability.
    1.  Regime Detection: Adds a "Ranging Market" regime detected by Bollinger
        Bandwidth. In this mode, the strategy switches from trend-following to
        mean-reversion to avoid whipsaws in sideways markets.
    2.  Enhanced Trend Indicators: Replaces Simple Moving Averages (SMA) with
        Exponential Moving Averages (EMA) for faster response to trend changes.
    3.  Dynamic Risk Management: The fixed percentage stop-loss is replaced with
        a more adaptive ATR-based trailing stop, which adjusts the risk threshold
        based on current market volatility.
    4.  Sentiment Refinement: The keyword dictionary is updated with more
        contemporary and nuanced terms like "supply chain" and "AI regulation".
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "technological breakthrough": 3.0, "short squeeze": 3.5, "capitulation": 3.0,
        # Ambiguous
        "strong jobs report": 0.5, "moderate growth": 0.5,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "supply chain disruption": -2.5, "ai regulation": -2.0,
        # Contrarian Negative (Extreme greed/euphoria can be a sell signal)
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
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
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 20
    BB_PERIOD = 20
    STOP_LOOKBACK = 20

    required_history_length = max(EMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_series_long = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_series_medium = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    ema_100 = ema_series_long[-1] if len(ema_series_long) > 0 else None
    ema_50 = ema_series_medium[-1] if len(ema_series_medium) > 0 else None
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOOKBACK:]) if len(all_prices) >= STOP_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, atr, bb_upper, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    # Crisis Regime: High-risk, high-volatility downtrend.
    is_long_term_downtrend = current_price < ema_100
    is_crisis_regime = is_long_term_downtrend and (current_price < ema_50)

    # Ranging Regime: Low volatility, sideways movement.
    bb_width = (bb_upper - bb_lower) / bb_middle
    is_ranging_market = bb_width < 0.06 # Threshold for a "squeeze" or low-volatility range

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION (HIGHEST PRIORITY)
    # If in a clear, long-term downtrend, the primary goal is capital preservation.
    if is_crisis_regime:
        return "SELL" # Exit all positions and wait for trend to recover.

    # REGIME 2: RANGING MARKET (MEAN REVERSION)
    # If the market is sideways, switch from trend-following to mean-reversion.
    if is_ranging_market:
        if current_price <= bb_lower and rsi < 35:
            return "BUY" # Buy near the bottom of the range
        if current_price >= bb_upper and rsi > 65:
            return "SELL" # Sell near the top of the range
        return "HOLD"

    # REGIME 3: NORMAL TRENDING MARKET
    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Volatility-Based Stop-Loss (ATR Trailing Stop)
    # Sell if price drops more than 2.5x ATR from the 20-day high.
    if current_price < (donchian_high_20 - 2.5 * atr):
        return "SELL"

    # Priority 2: Trend Breakdown Signal
    is_trend_reversing_down = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_reversing_down and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on Extreme Overbought Conditions
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > ema_50 and ema_50 > ema_100
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive = net_sentiment_score > -2.0 # Avoid buying into very negative news

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"