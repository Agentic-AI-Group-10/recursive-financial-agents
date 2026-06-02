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
        # Using pandas is preferred for accuracy and standard implementation
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback pure Python/Numpy implementation
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[0] = data_arr[0] # Start with the first price
        multiplier = 2 / (period + 1)
        for i in range(1, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
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
    
    ema_short = calculate_ema_series(prices, short_period)
    ema_long = calculate_ema_series(prices, long_period)
    
    macd_line = ema_short[-len(ema_long):] - ema_long
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[-len(signal_line):] - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    
    try:
        import pandas as pd
        atr_series = pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        return atr_series[-1] if len(atr_series) > 0 else None
    except ImportError:
        # Fallback calculation if pandas is not available
        atr = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr = (atr * (period - 1) + price_ranges[i]) / period
        return atr

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V2:
    This version evolves the parent by introducing more adaptive mechanisms:
    1.  Volatility-Adjusted Stop-Loss: Replaces the fixed 8% stop-loss with a
        dynamic level based on a multiple of the Average True Range (ATR). This
        adapts risk to current market volatility (wider stops in volatile times).
    2.  Sideways Market Filter: Explicitly identifies and filters out choppy,
        non-trending markets using an ATR ratio. This is designed to reduce
        whipsaw trades and conserve capital for high-conviction trends.
    3.  Refined Sentiment Logic: Adds a contrarian "euphoria" check to sell into
        extreme positive sentiment, acting as a profit-taking signal.
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
        "uncertainty": -1.5, "strong jobs report": -1.0,
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

    # Indicator Parameters
    SMA_LONG = 100
    SMA_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_CRASH_PERIOD = 20
    STOP_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0 # Multiple of ATR for stop-loss

    required_history_length = max(SMA_LONG + 1, ATR_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_LONG)
    sma_50 = calculate_sma(all_prices, SMA_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOOKBACK:]) if len(all_prices) >= STOP_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_hist = macd_hist_series[-1]
    prev_macd_hist = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    # Crisis Regime: Detects high-velocity crashes or sustained, volatile downtrends.
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (current_price < sma_100 and short_atr > (long_atr * 1.75)) or is_crash_velocity

    # Sideways Regime: Detects low-volatility, choppy markets to avoid whipsaws.
    is_low_vol_expansion = short_atr < (long_atr * 1.2) # Short-term vol not expanding vs long-term
    is_price_hugging_ma = abs(current_price - sma_50) / sma_50 < 0.03 # Price is within 3% of 50-day SMA
    is_sideways_regime = is_low_vol_expansion and is_price_hugging_ma

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION (Highest Priority)
    if is_crisis_regime:
        return "SELL" # Exit all positions immediately in a crisis.

    # REGIME 2: SIDEWAYS MARKET (Second Priority)
    if is_sideways_regime:
        return "HOLD" # Do not trade; wait for a clear trend to emerge.

    # REGIME 3: NORMAL TRENDING MARKET
    
    # --- SELL LOGIC ---
    # Priority 1: Volatility-Adjusted Trailing Stop-Loss.
    stop_loss_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * short_atr)
    if current_price < stop_loss_price:
        return "SELL"

    # Priority 2: Trend and Momentum Breakdown.
    is_trend_down = current_price < sma_50
    is_momentum_negative_cross = macd_hist < 0 and prev_macd_hist >= 0
    if is_trend_down and is_momentum_negative_cross:
        return "SELL"

    # Priority 3: Contrarian sell on euphoria / extreme overbought conditions.
    is_euphoric = rsi > 82 and net_sentiment_score > 4.0
    is_momentum_stalling = macd_hist > 0 and macd_hist < prev_macd_hist
    if is_euphoric and is_momentum_stalling:
        return "SELL"

    # --- BUY LOGIC ---
    is_trend_up = current_price > sma_50 and current_price > sma_100
    is_momentum_positive_cross = macd_hist > 0 and prev_macd_hist <= 0
    is_not_overbought = rsi < 75
    is_sentiment_supportive = net_sentiment_score > -2.0

    if is_trend_up and is_momentum_positive_cross and is_not_overbought and is_sentiment_supportive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"