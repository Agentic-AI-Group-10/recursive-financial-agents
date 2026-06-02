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
        # Using pandas is more robust and standard for financial calculations
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback implementation if pandas is not available
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[0] = data_arr[0]
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
    
    macd_line = ema_short_full - ema_long_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    
    # Use the EMA series function for a smoothed ATR
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return (prices[-1] - prices[-1 - period]) / prices[-1 - period]

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy incorporating a dynamic trailing stop-loss and a more
    responsive crisis detection system using a velocity indicator (ROC).
    1. Crisis Aversion: Enhanced to detect both high-volatility and high-velocity crashes.
    2. Normal Trend-Following: Adds a trailing stop-loss to protect profits and prevent
       "slow bleed" drawdowns, a key lesson from past successful agents.
    """
    # --- 1. Sentiment Analysis (Refined based on lessons) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -4.0, # Increased weight
        "quantitative tightening": -2.5, "black swan": -4.0, "systemic risk": -4.0,
        "contagion": -3.5, "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5,
        "geopolitical risk": -2.5, "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0,
        "bubble": -2.0, "uncertainty": -1.5, "strong jobs report": -0.5, # Reduced negative weight
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
    SMA_LONG = 100
    SMA_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_PERIOD = 20
    TRAILING_STOP_PERCENT = 0.08 # 8% drawdown from peak

    required_history_length = max(SMA_LONG + 1, ATR_LONG + 1, ROC_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_LONG)
    sma_50 = calculate_sma(all_prices, SMA_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)

    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection (Enhanced with Velocity Trigger) ---
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -0.12 # A sharp 12% drop in the last month
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL" # Exit any position immediately
        return "HOLD" # Stay in cash and wait

    # REGIME 2: NORMAL / RECOVERY (Unified Trend-Following with Trailing Stop)
    
    # --- SELL LOGIC (Layered for Robustness) ---
    # LAYER 1: DYNAMIC TRAILING STOP-LOSS (NEW)
    # To implement this without state, we find the last hypothetical buy signal and track the peak since.
    last_buy_index = -1
    # Search backwards for the last valid buy signal (MACD cross above SMA50)
    # We only need to check a reasonable lookback period, e.g., 252 trading days (1 year)
    search_start_index = max(required_history_length, len(all_prices) - 252)
    for i in range(len(all_prices) - 2, search_start_index, -1):
        # Check conditions at point `i`
        hist_slice = all_prices[:i+1]
        if calculate_sma(hist_slice, SMA_MEDIUM) is None: continue
        
        is_uptrend_then = hist_slice[-1] > calculate_sma(hist_slice, SMA_MEDIUM)
        
        # Check MACD cross at point `i`
        _, _, temp_macd_hist = calculate_macd_series(hist_slice)
        if temp_macd_hist is not None and len(temp_macd_hist) >= 2:
            if temp_macd_hist[-1] > 0 and temp_macd_hist[-2] <= 0:
                if is_uptrend_then:
                    last_buy_index = i
                    break # Found the most recent entry point
    
    if last_buy_index != -1:
        prices_since_buy = all_prices[last_buy_index:]
        recent_peak = max(prices_since_buy)
        if current_price < recent_peak * (1 - TRAILING_STOP_PERCENT):
            return "SELL"

    # LAYER 2: TREND BREAKDOWN
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_primary_downtrend and is_momentum_confirming_down:
        return "SELL"

    # LAYER 3: EXTREME OVERBOUGHT & FADING MOMENTUM
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Entry & Re-entry) ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.5

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"