import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_sma_series(data, period):
    """Calculates a full series of Simple Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).rolling(window=period).mean().to_numpy()[period-1:]
    except ImportError:
        weights = np.repeat(1.0, period) / period
        return np.convolve(data_arr, weights, 'valid')

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

def calculate_roc(prices, period):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return (prices[-1] - prices[-1 - period]) / prices[-1 - period]

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY:
    This version enhances the successful parent by introducing two key refinements:
    1. Choppy Market Filter: Uses the slope of the 50-day SMA to detect sideways markets,
       preventing whipsaw trades by holding during periods of low trend strength.
    2. Enhanced Crisis Detection: Adds a Rate-of-Change (ROC) check to identify
       high-velocity crashes, making the crisis aversion mode more responsive.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "strong jobs report": -1.0, # Reduced weight to be less impactful
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
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_CRASH_PERIOD = 10
    MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9

    required_history_length = max(SMA_TREND_LONG + 1, ATR_LONG + 1, MACD_LONG + MACD_SIGNAL, SMA_TREND_MEDIUM + 5)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50_series = calculate_sma_series(all_prices, SMA_TREND_MEDIUM)
    sma_50 = sma_50_series[-1]
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_10 = calculate_roc(all_prices, ROC_CRASH_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_10]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    # ENHANCED Crisis Detection: Adds a velocity component (ROC) to the parent's logic.
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.8) # Increased threshold for higher confidence
    is_crash_velocity = roc_10 < -0.08 # A drop of 8% in 10 days signals panic
    is_crisis_regime = is_long_term_downtrend and (is_high_volatility or is_crash_velocity)

    # NEW Choppy Market Filter: Detects sideways markets to avoid whipsaws.
    sma_50_slope = (sma_50_series[-1] - sma_50_series[-6]) / 5
    is_choppy_market = abs(sma_50_slope) < (sma_50 * 0.001) # Trend is moving less than 0.1% per day

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION (Highest Priority)
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD" # Hold cash and wait for the crisis to pass

    # REGIME 2: CHOPPY MARKET FILTER (Second Priority)
    # If the market is directionless, we avoid opening new trend-following positions.
    if is_choppy_market:
        # Exception: Allow profit-taking exits even in choppy markets.
        is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
        is_extremely_overbought = rsi > 80
        if is_extremely_overbought and is_momentum_fading:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL TREND-FOLLOWING (Default)
    is_primary_uptrend = current_price > sma_50

    # --- BUY LOGIC ---
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.5

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # --- SELL LOGIC ---
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.5

    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Additional profit-taking logic (also checked in choppy regime)
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    return "HOLD"