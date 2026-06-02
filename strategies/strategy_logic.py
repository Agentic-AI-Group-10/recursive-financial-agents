import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators (Retained for Robustness) ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        # Using pandas is preferred for accuracy and standard implementation
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback implementation if pandas is not available
        ema_values = np.zeros(len(data_arr), dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    
    if period == 0: return None # Avoid division by zero
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
    
    ema_short_full = calculate_ema_series(prices, short_period)
    ema_long_full = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter EMA series
    macd_line = ema_short_full[len(ema_short_full)-len(ema_long_full):] - ema_long_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    # Align histogram with the signal line
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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V2:
    This version evolves the parent strategy by introducing a more robust, score-based
    decision engine, replacing the previous brittle, rule-based logic.
    1.  Safety Overrides Retained: The successful crisis detection (ROC & Volatility)
        and dynamic stop-loss (Donchian High) are preserved as non-negotiable
        capital preservation rules that override the core signal.
    2.  Score-Based Signal Generation: Instead of a simple 'AND' logic chain, this
        version calculates a Bull and Bear score by aggregating evidence from multiple
        factors (trend, momentum, sentiment). A trade is only triggered if there is
        a strong confluence of signals, reducing whipsaws.
    3.  Faster Trend Indicators: Simple Moving Averages (SMAs) have been replaced
        with Exponential Moving Averages (EMAs) to provide more responsive trend signals.
    """
    # --- 1. Sentiment Analysis (Refined Keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive Keywords
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "disinflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5,
        "productivity boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "record high": 2.0,
        "strong earnings": 2.0, "strong guidance": 2.0, "beat estimates": 1.5,
        "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0, "capitulation": 3.0,
        # Negative Keywords
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "liquidity crisis": -3.5, "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5,
        "geopolitical risk": -2.5, "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0, "earnings miss": -2.5,
        "guidance cut": -2.5, "supply chain disruption": -2.0, "market turmoil": -2.0,
        "bubble": -2.0, "uncertainty": -1.5, "strong jobs report": -1.0,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0,
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
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = EMA_TREND_LONG + 10 # Safe buffer for all calculations
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [rsi, short_atr, long_atr, roc_20]) or \
       len(ema_100_series) == 0 or len(ema_50_series) == 0 or \
       macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    ema_100 = ema_100_series[-1]
    ema_50 = ema_50_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. High-Priority Overrides (Capital Preservation) ---

    # OVERRIDE 1: CRISIS AVERSION. Exit all positions in a crash.
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = short_atr > (long_atr * 1.8) # Stricter volatility threshold
    is_crash_velocity = roc_20 < -15.0
    if is_crash_velocity or (is_long_term_downtrend and is_high_volatility):
        return "SELL"

    # OVERRIDE 2: DYNAMIC STOP-LOSS. Protect profits or cut losses.
    if current_price < (donchian_high_20 * 0.92): # Sell if price drops 8% from 20-day high
        return "SELL"

    # --- 4. Score-Based Decision Engine (Normal/Trending Regime) ---
    bull_score = 0
    bear_score = 0

    # Factor 1: Trend (Weight: 2)
    if current_price > ema_50: bull_score += 1
    else: bear_score += 1
    if current_price > ema_100: bull_score += 1
    else: bear_score += 1

    # Factor 2: Momentum (Weight: 2)
    if macd_histogram > 0 and prev_macd_histogram <= 0: bull_score += 2 # Strong buy signal
    if macd_histogram < 0 and prev_macd_histogram >= 0: bear_score += 2 # Strong sell signal
    if macd_histogram > 0 and macd_histogram < prev_macd_histogram: bear_score += 0.5 # Fading bull momentum
    if macd_histogram < 0 and macd_histogram > prev_macd_histogram: bull_score += 0.5 # Fading bear momentum

    # Factor 3: Overbought/Oversold (Weight: 1)
    if rsi > 80: bear_score += 1 # Overbought, potential reversal
    if rsi < 30: bull_score += 1 # Oversold, potential bounce

    # Factor 4: Sentiment (Weight: 1)
    if net_sentiment_score >= 2.0: bull_score += 1
    if net_sentiment_score <= -2.0: bear_score += 1

    # Factor 5: Volatility Filter (Weight: 1) - Avoids choppy markets
    is_sufficient_volatility = short_atr > (long_atr * 0.7)
    if is_sufficient_volatility:
        bull_score += 1 # Only enable buying in active markets
    else:
        # In low volatility, we are more inclined to sell or hold
        bear_score += 0.5

    # --- 5. Final Decision ---
    BUY_THRESHOLD = 4.0
    SELL_THRESHOLD = 4.0

    if bull_score >= BUY_THRESHOLD and bear_score < 2.0:
        return "BUY"

    if bear_score >= SELL_THRESHOLD and bull_score < 2.0:
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"