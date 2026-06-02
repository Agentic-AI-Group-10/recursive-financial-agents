import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """
    Calculates a full series of Exponential Moving Averages.
    Returns a numpy array of EMA values.
    """
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        # Use pandas if available for a fast, standard implementation
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback to a numpy implementation if pandas is not installed
        ema_values = np.zeros(len(data_arr), dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) using Wilder's smoothing method.
    """
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
    """
    Calculates the MACD line, signal line, and histogram series.
    """
    if len(prices) < long_period:
        return None, None, None
        
    short_ema_full_series = calculate_ema_series(prices, short_period)
    long_ema_full_series = calculate_ema_series(prices, long_period)
    
    # Align the series by taking the tail of the shorter period's EMA
    macd_line = short_ema_full_series[-len(long_ema_full_series):] - long_ema_full_series
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line_full_series = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram with the signal line
    histogram = macd_line[-len(signal_line_full_series):] - signal_line_full_series
    
    return macd_line, signal_line_full_series, histogram

def calculate_atr(prices, period=14):
    """
    Calculates Average True Range (ATR) using close-to-close volatility.
    A simplified ATR for this context; a full implementation would use high/low prices.
    """
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """
    Calculates the Rate of Change (ROC) over a given period.
    """
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V2:
    This version evolves the successful parent by focusing on adaptability and signal confirmation.
    1.  EMA-Based Trend: Replaces SMAs with EMAs for faster reaction to trend changes.
    2.  ATR-Based Dynamic Stop-Loss: Replaces the static percentage stop-loss with a
        volatility-adjusted trailing stop (Donchian High - N*ATR) to reduce whipsaws.
    3.  RSI-Confirmed Entries: Adds an RSI > 52 filter to the BUY logic, ensuring
        entries have confirmed bullish momentum beyond a simple MACD crossover.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "peace talks": 2.0, "de-escalation": 2.0, "beat estimates": 1.5, "recovery": 1.5,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "yield curve inversion": -4.0, "black swan": -4.0, "systemic risk": -4.0,
        "contagion": -3.5, "recession": -3.0, "crisis": -3.0, "stagflation": -3.0,
        "hot inflation": -3.0, "war": -3.0, "quantitative tightening": -2.5,
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
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_VOL_PERIOD = 14
    ROC_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0

    required_history_length = max(EMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    ema_100 = ema_100_series[-1] if len(ema_100_series) > 0 else None
    ema_50 = ema_50_series[-1] if len(ema_50_series) > 0 else None
    
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_VOL_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < ema_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION
    if is_crisis_regime:
        # In a crisis, the primary goal is capital preservation. Sell if momentum is negative or trend breaks down.
        if macd_histogram < 0 or current_price < ema_50:
            return "SELL"
        return "HOLD" # Otherwise, hold cash and wait for a clear recovery signal.

    # REGIME 2: NORMAL / TREND-FOLLOWING

    # --- SELL LOGIC (Enhanced with Dynamic ATR Stop-Loss) ---
    # Priority 1: Dynamic Trailing Stop-Loss. Adapts to volatility.
    stop_loss_level = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < stop_loss_level:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_medium_term_downtrend = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.5
    if is_medium_term_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with fading momentum.
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Enhanced with RSI Momentum Confirmation) ---
    is_medium_term_uptrend = current_price > ema_50
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_momentum_confirmed_up = rsi > 52 # RSI must be in bullish territory to confirm entry
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.5

    if is_medium_term_uptrend and is_momentum_crossing_up and is_momentum_confirmed_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"