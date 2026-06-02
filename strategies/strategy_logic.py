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
    prices_arr = np.array(prices, dtype=float)
    short_ema_full = calculate_ema_series(prices_arr, short_period)
    long_ema_full = calculate_ema_series(prices_arr, long_period)
    macd_line = short_ema_full - long_ema_full
    if len(macd_line) < long_period + signal_period - 1: # Ensure enough data for signal line
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line[long_period-1:], signal_period)
    histogram = macd_line[long_period-1:][len(macd_line[long_period-1:])-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, high_prices, low_prices, period=14):
    """Calculates Average True Range (ATR). Uses close prices as a proxy for H/L if not available."""
    if len(prices) < period + 1:
        return None
    # Using close-to-close volatility as a robust proxy since H/L are not provided
    price_ranges = np.abs(np.diff(np.array(prices, dtype=float)))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bbands(prices, period=20, std_dev=2.0):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    prices_arr = np.array(prices, dtype=float)
    sma = np.mean(prices_arr[-period:])
    std = np.std(prices_arr[-period:])
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a more adaptive, regime-aware framework.
    1.  Adaptive Risk Management: Replaces the fixed stop-loss with a
        volatility-based ATR Chandelier Exit, which dynamically adjusts to
        market conditions, protecting profits without being prematurely stopped out.
    2.  Enhanced Regime Detection: Incorporates Bollinger Bands to gauge
        volatility and define market regimes (trending vs. crisis). This prevents
        trend-following signals in inappropriate, high-risk environments.
    3.  Improved Trend Identification: Upgrades from SMA to faster-reacting EMAs
        (21, 50, 200) for more timely and accurate trend assessment, forming the
        core of the trend-following logic.
    """
    # --- 1. Sentiment Analysis (Unchanged from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
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
    EMA_SHORT, EMA_MEDIUM, EMA_LONG = 21, 50, 200
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BBANDS_PERIOD = 20
    CHANDELIER_LOOKBACK = 22
    CHANDELIER_MULTIPLIER = 2.5

    required_history_length = EMA_LONG + 5 # Ensure enough data for all calculations
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_21_series = calculate_ema_series(all_prices, EMA_SHORT)
    ema_50_series = calculate_ema_series(all_prices, EMA_MEDIUM)
    ema_200_series = calculate_ema_series(all_prices, EMA_LONG)
    ema_21, ema_50, ema_200 = ema_21_series[-1], ema_50_series[-1], ema_200_series[-1]
    
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, None, None, ATR_PERIOD) # Proxy ATR with close prices
    bb_upper, bb_mid, bb_lower = calculate_bbands(all_prices, BBANDS_PERIOD)

    # Calculate Chandelier Exit (stateless version)
    highest_high_lookback = np.max(all_prices[-CHANDELIER_LOOKBACK:])
    chandelier_exit = highest_high_lookback - (atr * CHANDELIER_MULTIPLIER) if atr else None

    # Null check for all indicators
    if any(v is None for v in [ema_21, ema_50, ema_200, rsi, atr, bb_upper, chandelier_exit]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_uptrend = current_price > ema_200
    is_medium_term_uptrend = ema_21 > ema_50
    
    bb_width = ((bb_upper - bb_lower) / bb_mid) * 100 if bb_mid > 0 else 0
    is_high_volatility = bb_width > 10.0 # 10% width indicates significant volatility
    
    is_crisis_regime = not is_long_term_uptrend and is_high_volatility

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: RISK MANAGEMENT (HIGHEST PRIORITY SELL)
    # Priority 1: ATR-based Chandelier Exit for dynamic stop-loss.
    if current_price < chandelier_exit:
        return "SELL"

    # Priority 2: Crisis Aversion. Exit if in a crisis and momentum is negative.
    if is_crisis_regime and macd_histogram < 0:
        return "SELL"

    # REGIME 2: CONTRARIAN BUY (High-conviction reversal)
    is_deeply_oversold = rsi < 22
    is_price_at_band = current_price < bb_lower
    is_momentum_turning_up = macd_hist_delta > 0 and macd_histogram < 0 # Must be turning from negative
    
    if is_deeply_oversold and is_price_at_band and is_momentum_turning_up and not is_crisis_regime:
        return "BUY"

    # REGIME 3: TREND FOLLOWING (Normal Market Conditions)
    if not is_crisis_regime:
        # --- SELL LOGIC (Profit-taking / Trend breakdown) ---
        is_trend_breakdown = ema_21 < ema_50 and ema_21_series[-2] >= ema_50_series[-2]
        is_momentum_breakdown = macd_histogram < 0 and prev_macd_histogram >= 0
        if is_trend_breakdown and is_momentum_breakdown:
            return "SELL"

        is_extremely_overbought = rsi > 80
        is_momentum_fading = macd_hist_delta < 0
        if is_extremely_overbought and is_momentum_fading and current_price > bb_upper:
            return "SELL"

        # --- BUY LOGIC ---
        is_pullback_in_uptrend = is_long_term_uptrend and is_medium_term_uptrend and current_price > ema_50
        is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
        is_not_overextended = rsi < 75 and current_price < bb_upper * 1.02 # Avoid buying blow-off tops
        is_sentiment_permissive = net_sentiment_score > -2.5

        if is_pullback_in_uptrend and is_momentum_confirming_up and is_not_overextended and is_sentiment_permissive:
            return "BUY"

    # Default action is to hold the current position.
    return "HOLD"