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
        # Use pandas if available for a robust, standard implementation
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback to a manual calculation if pandas is not installed
        ema_values = np.zeros_like(data_arr)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
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
    
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    macd_line = full_short_ema - full_long_ema
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    # Return only the valid, non-zero parts of the series
    return macd_line[long_period-1:], signal_line[long_period+signal_period-2:], histogram[long_period+signal_period-2:]


def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use the more robust full series calculation for ATR
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy based on a unified "Market Health" framework. It prioritizes
    capital preservation in high-risk environments and uses a high-conviction "Phoenix"
    signal for re-entry after a crisis, while employing robust trend-following in
    normal conditions.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Retained from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-impact Macro (Positive)
        "fed pivot": 4.0, "quantitative easing": 3.5, "stimulus": 3.0, "rate cut": 3.0,
        "soft landing": 2.5, "cooling inflation": 2.5, "cpi miss": 2.5, "dovish": 2.0,
        # High-impact Macro (Negative)
        "black swan": -5.0, "systemic risk": -4.5, "yield curve inversion": -4.0, "crisis": -4.0,
        "recession": -3.5, "stagflation": -3.5, "hot inflation": -3.5, "war": -3.5,
        "quantitative tightening": -3.0, "rate hike": -3.0, "credit crunch": -3.0,
        # Contrarian (used with caution)
        "capitulation": 2.0, "panic selling": 1.5, "extreme fear": 1.0,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0,
        # Standard News
        "strong earnings": 2.0, "bullish": 1.5, "breakthrough": 2.0, "record high": 1.5,
        "hawkish": -2.0, "bearish": -1.5, "plunge": -2.0, "sell-off": -1.5, "uncertainty": -1.0,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Market Health Assessment ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RECOVERY_SMA_PERIOD = 20  # For Phoenix signal
    TREND_SMA_PERIOD = 100    # Core Market Health filter
    RSI_PERIOD = 14
    ATR_SHORT_PERIOD = 10
    ATR_LONG_PERIOD = 50

    required_history_length = max(TREND_SMA_PERIOD + 1, ATR_LONG_PERIOD + 1, LONG_EMA_PERIOD + 9)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate all indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    sma_20 = calculate_sma(all_prices, RECOVERY_SMA_PERIOD)
    sma_100 = calculate_sma(all_prices, TREND_SMA_PERIOD)
    sma_100_yesterday = calculate_sma(all_prices[:-1], TREND_SMA_PERIOD)
    
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    
    short_atr = calculate_atr(all_prices, ATR_SHORT_PERIOD)
    long_atr = calculate_atr(all_prices, ATR_LONG_PERIOD)
    
    _, _, macd_hist_series = calculate_macd_series(all_prices)

    # Robustness check for all calculations
    if any(v is None for v in [short_ema, long_ema, sma_20, sma_100, sma_100_yesterday, rsi, short_atr, long_atr]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_hist = macd_hist_series[-1]
    prev_macd_hist = macd_hist_series[-2]

    # --- 3. Unified Regime Definition (Market Health) ---
    # This is the core defensive trigger, addressing the failure to act early in a crash.
    is_high_volatility = short_atr > (long_atr * 1.8)
    is_long_term_trend_broken = current_price < sma_100
    IS_RISK_OFF_ENVIRONMENT = is_long_term_trend_broken and is_high_volatility

    # --- 4. Decision Logic ---

    # === REGIME 1: RISK-OFF / CRASH PROTECTION ===
    # If the market is unhealthy, the only goal is capital preservation. No buying.
    if IS_RISK_OFF_ENVIRONMENT:
        return "SELL" # Liquidate any holdings and stay in cash.

    # === REGIME 2: PHOENIX RE-ENTRY (Post-Crash Recovery) ===
    # This logic only runs if the market is NO LONGER in a Risk-Off state.
    # It looks for a high-conviction signal to re-enter after a crisis.
    was_recently_risk_off = price_history[-1] < sma_100_yesterday
    if was_recently_risk_off:
        # The "Phoenix Signal" requires a confluence of strong evidence for recovery.
        is_price_recovering = current_price > sma_20
        is_momentum_reversing = macd_hist > 0 and prev_macd_hist <= 0
        has_macro_support = net_sentiment_score > 3.0 # High bar for stimulus/pivot news
        
        if is_price_recovering and is_momentum_reversing and has_macro_support:
            return "BUY"
        else:
            return "HOLD" # Wait for full confirmation; don't jump back in too early.

    # === REGIME 3: NORMAL TREND-FOLLOWING (Risk-On Environment) ===
    # Simplified and robust logic for healthy market conditions.
    is_bullish_trend = short_ema > long_ema
    is_momentum_positive = macd_hist > 0
    is_strength_confirmed = rsi > 52 # Use RSI > 50 as a basic strength filter

    # BUY Condition: Enter when trend, momentum, and strength align.
    if is_bullish_trend and is_momentum_positive and is_strength_confirmed:
        return "BUY"

    # SELL Condition: Exit when the trend fails or shows signs of exhaustion.
    # This is more tolerant than the parent's tight stop-loss, allowing trends to run.
    is_trend_failing = short_ema < long_ema
    is_overbought_and_fading = rsi > 78 and macd_hist < prev_macd_hist
    
    if is_trend_failing or is_overbought_and_fading:
        return "SELL"

    # Default action is to hold the current position if no strong signal is present.
    return "HOLD"