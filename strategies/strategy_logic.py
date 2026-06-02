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
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

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
    
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    # Align the series by taking the tail of the shorter period EMA
    macd_line = full_short_ema[long_period-short_period:] - full_long_ema

    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram with the end of the signal line
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use EMA series for ATR calculation
    try:
        import pandas as pd
        return pd.Series(price_ranges).ewm(span=period, adjust=False).mean().iloc[-1]
    except ImportError:
        # Fallback if pandas is not available
        atr_val = np.mean(price_ranges[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(price_ranges)):
            atr_val = (price_ranges[i] - atr_val) * multiplier + atr_val
        return atr_val

def calculate_roc(prices, period=10):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY:
    This version enhances the successful parent by:
    1.  Upgrading SMAs to more responsive EMAs to reduce indicator lag.
    2.  Adding a Rate-of-Change (ROC) indicator to the crisis detection module to
        react faster to velocity-driven flash crashes.
    3.  Introducing a tighter profit-taking/risk-management exit based on a
        short-term EMA (20-day) to prevent "slow bleed" drawdowns on winning trades.
    4.  Refining sentiment keywords for more nuance.
    """
    # --- 1. Sentiment Analysis (Refined) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 2.5,
        # Contrarian Positive
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "deleveraging": -2.5, "uncertainty": -1.5, "strong jobs report": -1.0, # Muted negative
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
    EMA_TREND_SHORT = 20 # For tighter exits
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_PERIOD = 10
    MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9

    required_history_length = max(EMA_TREND_LONG + 1, ATR_LONG + 1, MACD_LONG + MACD_SIGNAL)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100 = calculate_ema(all_prices, EMA_TREND_LONG)
    ema_50 = calculate_ema(all_prices, EMA_TREND_MEDIUM)
    ema_20 = calculate_ema(all_prices, EMA_TREND_SHORT)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_10 = calculate_roc(all_prices, ROC_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, ema_20, rsi, short_atr, long_atr, roc_10]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection (Enhanced for Velocity) ---
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = short_atr > (long_atr * 1.8) # Slightly higher threshold
    is_flash_crash = roc_10 < -10.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_flash_crash

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION (More Decisive)
    # If a crisis is detected (either structural or velocity-based), the immediate
    # priority is capital preservation. Exit all positions.
    if is_crisis_regime:
        return "SELL"

    # REGIME 2: NORMAL / RECOVERY (Unified Trend-Following with Tighter Risk Management)
    is_uptrend_structure = current_price > ema_50 and ema_50 > ema_100

    # --- BUY LOGIC (Entry & Re-entry) ---
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75 # Tighter entry threshold
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0 # Tighter sentiment filter

    if is_uptrend_structure and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # --- SELL LOGIC (Multiple Exit Conditions for Robustness) ---
    
    # A. Primary Trend Breakdown Exit
    is_trend_reversing = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_reversing and is_momentum_confirming_down:
        return "SELL"

    # B. Profit-Taking / Weakness Exit (Prevents "Slow Bleed")
    # If price violates the short-term trend and momentum is fading, exit to lock in gains.
    is_short_term_trend_broken = current_price < ema_20
    is_momentum_fading = macd_histogram < prev_macd_histogram
    if is_short_term_trend_broken and is_momentum_fading and macd_histogram > 0:
        return "SELL"

    # C. Exhaustion Exit
    is_extremely_overbought = rsi > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"