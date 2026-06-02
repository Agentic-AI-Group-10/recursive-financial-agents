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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        ema_values[0] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(1, len(ema_values)):
            ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

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
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    sma = np.mean(prices[-period:])
    std_dev = np.std(prices[-period:])
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version integrates Bollinger Bands for dynamic, volatility-based analysis,
    addressing a key weakness of fixed-threshold indicators.
    1.  Volatility Regime Filter: A new "low-volatility" regime, identified by
        narrow Bollinger Band Width, now filters out trend-following signals
        during choppy, sideways markets to reduce whipsaw losses.
    2.  Dynamic Confirmation Signals: Buy and sell signals are now confirmed by
        price action relative to the Bollinger Bands. For instance, a BUY requires
        the price to be above the middle band, and profit-taking is triggered when
        an overbought price falls back inside the upper band.
    3.  Enhanced Sentiment Dictionary: The keyword list is updated with more
        nuanced economic terms ("disinflation", "guidance cut") to better capture
        the current market narrative and improve the model's contextual awareness.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive Keywords
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "disinflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5,
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0,
        "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "supply chain easing": 2.0, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5, # Ambiguous
        # Negative Keywords
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "guidance cut": -2.5, "market turmoil": -2.0,
        "bubble": -2.0, "uncertainty": -1.5,
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
    yesterday_prices = price_history + [current_price] # Use yesterday's close for BB comparison

    # Indicator Periods
    SMA_TREND_LONG = 100
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    
    # Null check for all indicators
    if any(v is None for v in [sma_100, rsi, roc_20, donchian_high_20, bb_upper]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    # Crisis Regime: High-risk environment
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    # Capitulation Regime: Extreme subset of crisis, potential bottom
    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # Low Volatility (Ranging) Regime: Sideways market, avoid trend signals
    bollinger_band_width = (bb_upper - bb_lower) / bb_middle
    is_low_volatility_regime = bollinger_band_width < 0.04 # 4% width indicates a tight range

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < bb_middle:
            return "SELL"
        return "HOLD" # Stay defensive in cash

    # REGIME 3: LOW VOLATILITY / RANGING MARKET
    if is_low_volatility_regime:
        return "HOLD" # Avoid trading in choppy, directionless markets

    # REGIME 4: NORMAL TRENDING MARKET

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Stop-Loss (7% drop from 20-day high)
    if current_price < (donchian_high_20 * 0.93):
        return "SELL"

    # Priority 2: Trend Exhaustion / Mean Reversion
    is_overbought = rsi > 78
    is_momentum_fading = macd_hist_delta < 0
    price_reverted_from_peak = current_price < bb_upper and yesterday_prices[-2] >= bb_upper
    if is_overbought and is_momentum_fading and price_reverted_from_peak:
        return "SELL"

    # Priority 3: Standard Trend Breakdown Signal
    is_mid_term_downtrend = current_price < bb_middle
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.0
    if is_mid_term_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # --- BUY LOGIC ---
    is_mid_term_uptrend = current_price > bb_middle
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overextended = rsi < 75
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.5

    if is_mid_term_uptrend and is_momentum_confirming_up and is_not_overextended and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"