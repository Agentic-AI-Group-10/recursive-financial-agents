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
        ema_values = np.zeros_like(data_arr, dtype=float)
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
    # Ensure full series are calculated to align indices correctly
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    # MACD line calculation
    macd_line = full_short_ema - full_long_ema
    
    if len(prices) < long_period + signal_period:
        return macd_line, None, None
        
    # Signal line calculation on the MACD line
    signal_line = calculate_ema_series(macd_line[long_period-1:], signal_period)
    
    # Align histogram
    aligned_macd_line = macd_line[long_period-1+signal_period-1:]
    histogram = aligned_macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use EMA series for ATR calculation as is standard
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_chandelier_exit(prices, period=22, multiplier=3.0):
    """Calculates the Chandelier Exit (long) stop level."""
    if len(prices) < period:
        return None, None
    
    atr = calculate_atr(prices, period)
    if atr is None:
        return None, None
        
    highest_high = np.max(prices[-period:])
    stop_level = highest_high - (atr * multiplier)
    return stop_level, atr

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY:
    Retains the successful dual-regime model but enhances it with:
    1. EMA-based trend filters for faster response times.
    2. A Chandelier Exit (ATR-based trailing stop) to combat the "slow bleed"
       problem by providing dynamic risk management.
    3. A volatility filter on buy signals to avoid entering chaotic markets.
    4. Refined sentiment keywords for better cross-cycle robustness.
    """
    # --- 1. Sentiment Analysis (Refined) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "buybacks": 1.5, "strong jobs report": 1.0, # Adjusted from negative
        # Contrarian Positive
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "liquidity crisis": -4.0,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "credit crunch": -3.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "uncertainty": -1.5,
        # Contrarian Negative
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
    ATR_SHORT = 10
    ATR_LONG = 50
    CHANDELIER_PERIOD = 22
    MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9

    required_history_length = max(EMA_TREND_LONG + 1, ATR_LONG + 1, MACD_LONG + MACD_SIGNAL, CHANDELIER_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100 = calculate_ema(all_prices, EMA_TREND_LONG)
    ema_50 = calculate_ema(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    chandelier_stop, _ = calculate_chandelier_exit(all_prices, CHANDELIER_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, short_atr, long_atr, chandelier_stop]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crisis_regime = is_long_term_downtrend and is_high_volatility

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < ema_50:
            return "SELL"
        return "HOLD" # Preserve capital, do not buy dips

    # REGIME 2: NORMAL / RECOVERY
    is_primary_uptrend = current_price > ema_50

    # --- SELL LOGIC (Priority 1: Risk Management & Profit Taking) ---
    # NEW: Chandelier Exit to prevent "slow bleed" losses and lock in profits.
    if current_price < chandelier_stop:
        return "SELL"

    # Profit-taking for extreme overbought conditions with fading momentum.
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = rsi > 82 # Slightly higher threshold for more confidence
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"
        
    # Trend breakdown confirmation sell.
    is_primary_downtrend = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.5
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # --- BUY LOGIC (Entry & Re-entry) ---
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.5
    # NEW: Volatility filter to avoid buying into chaotic, unpredictable spikes.
    is_volatility_stable = short_atr < (long_atr * 1.6)

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_volatility_stable:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"