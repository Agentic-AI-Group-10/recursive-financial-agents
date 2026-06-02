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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_cmo(prices, period=14):
    """Calculates the Chande Momentum Oscillator (CMO)."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr[-period-1:])
    
    sum_up = np.sum([d for d in deltas if d > 0])
    sum_down = np.sum([abs(d) for d in deltas if d < 0])
    
    if (sum_up + sum_down) == 0:
        return 0.0
        
    return 100 * (sum_up - sum_down) / (sum_up + sum_down)

def calculate_efficiency_ratio(prices, period=20):
    """Calculates a market efficiency ratio to detect choppiness (0=trending, 1=choppy)."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices[-period-1:], dtype=float)
    net_change = abs(prices_arr[-1] - prices_arr[0])
    total_volatility = np.sum(np.abs(np.diff(prices_arr)))
    
    if total_volatility == 0:
        return 1.0 # No movement is max choppiness/no trend.
    efficiency = net_change / total_volatility
    return 1.0 - efficiency

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V2:
    This version evolves the successful parent with more adaptive mechanisms:
    1.  Adaptive Trailing Stop: Replaces the fixed percentage stop-loss with a
        dynamic, ATR-based trailing stop to better adapt to market volatility.
    2.  Choppiness Filter: Introduces a market efficiency ratio to identify
        sideways, whipsaw-prone markets, preventing low-conviction trades.
    3.  Enhanced Momentum Indicator: Integrates the Chande Momentum Oscillator (CMO)
        to complement existing indicators for a more robust view of market strength.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "disinflation": 2.5, "ai boom": 2.5,
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
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

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    CMO_PERIOD = 14
    ATR_PERIOD = 14
    ROC_PERIOD = 20
    STOP_LOOKBACK = 20
    CHOP_PERIOD = 20

    required_history_length = max(SMA_TREND_LONG + 1, CHOP_PERIOD + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    cmo = calculate_cmo(all_prices, CMO_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    choppiness = calculate_efficiency_ratio(all_prices, CHOP_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOOKBACK:]) if len(all_prices) >= STOP_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, cmo, atr, roc_20, choppiness, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_crash_velocity)
    is_choppy_market = choppiness > 0.65 # High value (near 1.0) means choppy

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION (Highest Priority)
    if is_crisis_regime:
        return "SELL"

    # --- SELL LOGIC (Exit Triggers) ---
    # Priority 1: Adaptive Trailing Stop-Loss (Chandelier Exit style)
    trailing_stop_price = donchian_high_20 - (atr * 2.5)
    if current_price < trailing_stop_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal
    is_trend_breakdown = current_price < sma_50
    is_momentum_crossing_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_not_bullish = net_sentiment_score < 3.0
    if is_trend_breakdown and is_momentum_crossing_down and is_sentiment_not_bullish:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with fading momentum
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = cmo > 65
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Entry Triggers) ---
    # Condition 1: Must not be in a choppy, directionless market
    if is_choppy_market:
        return "HOLD"

    # Condition 2: Must be in a primary uptrend with confirming momentum
    is_primary_uptrend = current_price > sma_50
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = cmo < 70
    is_sentiment_not_bearish = net_sentiment_score > -3.0

    if is_primary_uptrend and is_momentum_crossing_up and is_not_overbought and is_sentiment_not_bearish:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"