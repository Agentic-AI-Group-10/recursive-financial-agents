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
        # Using pandas is preferred for speed and accuracy
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback pure Python/Numpy implementation
        ema_values = np.zeros_like(data_arr, dtype=float)
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
    
    # Ensure we get full series aligned with original prices length
    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter one
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram
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
    SELF-IMPROVED STRATEGY v2:
    This version evolves the parent by implementing the following key upgrades:
    1.  EMA-Based Trend: Replaces SMAs with more responsive EMAs (50/200) for
        faster and more accurate trend identification.
    2.  Dynamic ATR Stop-Loss: The fixed-percentage stop-loss is replaced with a
        volatility-adaptive ATR trailing stop, improving risk management.
    3.  Choppy Market Filter: A new regime detection for low-volatility, sideways
        markets is introduced to prevent whipsaw trades and conserve capital.
    4.  Enhanced Sentiment Integration: Sentiment analysis is refined with more
        macroeconomic keywords and now directly modulates trade entry/exit thresholds.
    """
    # --- 1. Sentiment Analysis (Enhanced Keyword Set) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive Catalysts
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 3.0, "soft landing": 3.0,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ppi miss": 2.0, "ai breakthrough": 2.5,
        "stimulus package": 2.0, "dovish": 2.0, "bullish consensus": 2.0, "strong earnings": 2.0,
        "beats estimates": 1.5, "economic recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        # Contrarian Positive (Fear Capitulation)
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Negative Catalysts
        "recession confirmed": -4.0, "crisis": -4.0, "stagflation": -3.5, "hot inflation": -3.5,
        "geopolitical escalation": -3.5, "yield curve inversion": -3.5, "quantitative tightening": -3.0,
        "black swan": -5.0, "systemic risk": -5.0, "contagion": -4.0,
        "rate hike": -2.5, "unexpected hike": -3.5, "bankruptcy": -3.0, "hard landing": -3.0,
        "cpi beat": -2.5, "ppi beat": -2.0, "vix spike": -2.5, "hawkish": -2.0,
        "bear market": -2.0, "sell-off": -2.0, "weak guidance": -2.5, "market turmoil": -2.0,
        "asset bubble": -2.5, "uncertainty": -1.5,
        # Contrarian Negative (Greed Topping Signal)
        "euphoria": -3.0, "mania": -3.5, "irrational exuberance": -3.5, "extreme greed": -3.0,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "avoids", "prevent", "easing"]
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
    EMA_TREND_LONG = 200
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ROC_PERIOD = 20
    STOP_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0

    required_history_length = EMA_TREND_LONG + 1
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_200 = calculate_ema_series(all_prices, EMA_TREND_LONG)[-1]
    ema_50 = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)[-1]
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOOKBACK:]) if len(all_prices) >= STOP_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_200, ema_50, rsi, atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection (Expanded) ---
    is_long_term_downtrend = current_price < ema_200
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    is_price_hugging_ema = abs(current_price - ema_50) / ema_50 < 0.02 # Price is within 2% of EMA50
    is_volatility_contracting = atr < (np.mean([calculate_atr(all_prices, p) for p in [50, 100]]) * 0.8)
    is_choppy_regime = is_price_hugging_ema and is_volatility_contracting

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION (Highest Priority)
    if is_crisis_regime:
        # In a crisis, the primary goal is capital preservation. Sell on any sign of weakness.
        if current_price < ema_50 or macd_histogram < 0:
            return "SELL"
        return "HOLD" # Otherwise, hold cash and wait for a clear recovery signal.

    # --- SELL LOGIC (Dynamic Stop-Loss and Sentiment-Driven) ---
    # Priority 1: Dynamic ATR-based Trailing Stop-Loss.
    atr_stop_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < atr_stop_price:
        return "SELL"

    # Priority 2: Catastrophic News Event.
    is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
    if net_sentiment_score < -4.5 and is_momentum_fading:
        return "SELL"

    # Priority 3: Standard Trend Breakdown.
    is_medium_term_downtrend = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_medium_term_downtrend and is_momentum_confirming_down:
        return "SELL"
        
    # Priority 4: Profit-taking on extreme overbought conditions.
    rsi_sell_threshold = 82
    is_extremely_overbought = rsi > rsi_sell_threshold
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Regime-Aware) ---
    is_medium_term_uptrend = current_price > ema_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    
    # Modulate RSI buy threshold based on sentiment
    rsi_buy_threshold = 78
    if net_sentiment_score > 3.0: # Very positive news can justify buying into strength
        rsi_buy_threshold = 82
    is_not_overbought = rsi < rsi_buy_threshold

    # Define the base buy condition
    base_buy_signal = is_medium_term_uptrend and is_momentum_confirming_up and is_not_overbought

    # In a choppy market, require a stronger confirmation signal to avoid false breakouts.
    if is_choppy_regime:
        # Require price to be decisively above EMA and stronger momentum.
        is_stronger_breakout = current_price > (ema_50 * 1.01)
        is_stronger_momentum = macd_histogram > (atr * 0.05) # MACD hist > 5% of ATR
        if base_buy_signal and is_stronger_breakout and is_stronger_momentum:
            return "BUY"
    else: # Normal trending market
        if base_buy_signal:
            return "BUY"

    # Default action is to hold the current position.
    return "HOLD"