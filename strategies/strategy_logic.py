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

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band

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
    A self-improved, multi-regime trading strategy incorporating lessons from past
    failures. It uses a velocity-based crash detection system and a unified
    "Conviction Score" to reduce whipsaws and improve signal quality.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Unchanged from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "stimulus": 2.0, "dovish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "capitulation": 3.0, "panic selling": 2.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "black swan": -4.0, "systemic risk": -4.0,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "market turmoil": -2.0, "uncertainty": -1.5,
        "strong jobs report": -1.5, "euphoria": -2.5, "mania": -3.0,
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
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    LONG_TERM_SMA_PERIOD = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ROC_PERIOD = 10 # **NEW** For velocity detection

    required_history_length = max(LONG_EMA_PERIOD + 9, LONG_TERM_SMA_PERIOD + 1, ROC_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    sma_50 = calculate_sma(all_prices, LONG_TERM_SMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    prev_rsi = calculate_rsi(all_prices[:-1], RSI_PERIOD)
    middle_band, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    roc = calculate_roc(all_prices, ROC_PERIOD)

    if any(v is None for v in [short_ema, long_ema, sma_50, rsi, prev_rsi, middle_band, roc]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Enhanced Regime Detection (Addresses Indicator Lag) ---
    is_long_term_bearish = current_price < sma_50
    is_high_velocity_downturn = roc < -8.0 # Detects sharp drops quickly
    
    # **IMPROVEMENT**: Crash mode is now triggered by structure (below SMA) AND velocity (fast drop).
    # This is more responsive than ATR-based volatility, which lags.
    CRASH_MODE = is_long_term_bearish and is_high_velocity_downturn

    # --- 4. Multi-Regime Decision Logic ---
    if CRASH_MODE:
        # === CRASH PROTECTION MODE: Prioritize capital preservation and V-bottom hunting. ===
        # SELL signal is aggressive to exit failing positions immediately.
        if short_ema < long_ema and rsi < 50:
            return "SELL"

        # **IMPROVEMENT**: V-Bottom Hunter. This logic is designed to buy near the point of max panic,
        # addressing the failure of missing the COVID recovery. It uses RSI and ROC for speed.
        is_deeply_oversold = prev_rsi is not None and prev_rsi < 25 and rsi > prev_rsi
        is_reversing_sharply = roc > 5.0 # Look for a sharp bounce
        has_capitulation_news = net_sentiment_score > 2.0 # "panic selling", "capitulation"
        
        if is_deeply_oversold and is_reversing_sharply and has_capitulation_news:
            return "BUY"
        
        # Default action in a crash is to HOLD cash and wait for a clear signal.
        return "HOLD"

    # --- Logic for all other non-crash scenarios (NORMAL MODE) ---
    else:
        # **IMPROVEMENT**: Unified "Conviction Score" to replace complex if/else trees.
        # This reduces whipsaws and makes decisions more robust.
        
        # 1. Trend Score (-2 to +2)
        trend_score = 0
        if short_ema > long_ema: trend_score += 1
        if current_price > sma_50: trend_score += 1
        if short_ema < long_ema: trend_score -= 1
        if current_price < sma_50: trend_score -= 1
        
        # 2. Momentum Score (-2 to +2)
        momentum_score = 0
        if macd_histogram > 0 and macd_histogram > prev_macd_histogram: momentum_score += 1 # Accelerating up
        if rsi > 55: momentum_score += 1
        if macd_histogram < 0 and macd_histogram < prev_macd_histogram: momentum_score -= 1 # Accelerating down
        if rsi < 45: momentum_score -= 1

        # 3. Sentiment Score (normalized to approx -2 to +2)
        sentiment_score_scaled = np.clip(net_sentiment_score / 2.5, -2, 2)

        total_score = trend_score + momentum_score + sentiment_score_scaled

        # **IMPROVEMENT**: Exit logic to prevent "slow bleed".
        # If we are in a long position that is starting to fail, get out.
        is_momentum_fading = macd_histogram < prev_macd_histogram and macd_histogram > 0
        is_breaking_support = current_price < middle_band # Middle Bollinger Band
        if is_breaking_support and is_momentum_fading:
            return "SELL"

        # BUY Signal: High conviction score, but not at an extreme overbought level.
        if total_score >= 3.5 and rsi < 78:
            return "BUY"
            
        # SELL Signal: High negative conviction score, but not at an extreme oversold level.
        if total_score <= -3.5 and rsi > 22:
            return "SELL"

    return "HOLD"