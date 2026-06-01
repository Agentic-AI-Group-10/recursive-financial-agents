import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.zeros_like(prices_arr, dtype=float)
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values[-1]

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line and Signal line for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None
    
    prices_arr = np.array(prices, dtype=float)
    
    # Calculate Short EMA
    short_ema_values = np.zeros_like(prices_arr, dtype=float)
    short_ema_values[short_period - 1] = np.mean(prices_arr[:short_period])
    short_multiplier = 2 / (short_period + 1)
    for i in range(short_period, len(prices_arr)):
        short_ema_values[i] = (prices_arr[i] - short_ema_values[i-1]) * short_multiplier + short_ema_values[i-1]

    # Calculate Long EMA
    long_ema_values = np.zeros_like(prices_arr, dtype=float)
    long_ema_values[long_period - 1] = np.mean(prices_arr[:long_period])
    long_multiplier = 2 / (long_period + 1)
    for i in range(long_period, len(prices_arr)):
        long_ema_values[i] = (prices_arr[i] - long_ema_values[i-1]) * long_multiplier + long_ema_values[i-1]
        
    macd_line_values = short_ema_values - long_ema_values
    
    # Calculate Signal Line (EMA of MACD)
    signal_line_values = np.zeros_like(macd_line_values, dtype=float)
    macd_for_signal = macd_line_values[long_period-1:]
    if len(macd_for_signal) < signal_period:
        return None, None
        
    signal_line_values[long_period - 1 + signal_period - 1] = np.mean(macd_for_signal[:signal_period])
    signal_multiplier = 2 / (signal_period + 1)
    for i in range(long_period - 1 + signal_period, len(prices_arr)):
        signal_line_values[i] = (macd_line_values[i] - signal_line_values[i-1]) * signal_multiplier + signal_line_values[i-1]

    return macd_line_values[-1], signal_line_values[-1]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    if len(deltas) < period:
        return None
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
        
    rsi = 100 - (100 / (1 + rs))
    return rsi

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy addressing passivity by introducing more responsive
    exit signals using MACD and price-level breaks, while retaining the proven
    multi-regime (volatility-based) and multi-factor confirmation logic.

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
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + SIGNAL_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line]):
        return "HOLD"

    # Adaptive Volatility Regime (Retained from successful parent)
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Improved Multi-Regime Decision Logic with Faster Exits ---
    
    # Define technical state
    bullish_trend = short_ema > long_ema
    bullish_momentum = macd_line > signal_line
    price_above_support = current_price > long_ema

    # --- BUY Conditions (Entry Logic) ---
    # Conditions to enter a new long position
    buy_signal = False
    if is_high_volatility:
        # CRISIS MODE: Stricter entry
        if (bullish_trend and bullish_momentum and price_above_support and
            net_sentiment_score >= 2.0 and rsi < 65):
            buy_signal = True
    else:
        # NORMAL MODE: Standard entry
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market: # Normal Trending
            if (bullish_trend and bullish_momentum and
                net_sentiment_score >= 1.0 and rsi < 70):
                buy_signal = True
        else: # Choppy / Mean-Reversion
            if rsi < 25 and net_sentiment_score > -1.5:
                buy_signal = True

    # --- SELL Conditions (Crucial Improvement: Proactive Exit Logic) ---
    # Conditions to exit an existing long position
    sell_signal = False
    
    # A) Strong reversal signal (original logic, but now with MACD)
    strong_reversal = (not bullish_trend and not bullish_momentum and net_sentiment_score <= -1.0)
    
    # B) Protective Stop: Trend weakness detected (NEW, FASTER EXIT)
    # Exit if momentum dies OR price breaks key support
    trend_weakness = (not bullish_momentum or not price_above_support)
    
    # C) Extreme Overbought/Euphoria signal
    extreme_overbought = rsi > 78 and net_sentiment_score < 1.5
    
    # D) Panic Sell on catastrophic news
    panic_news = net_sentiment_score <= -3.0

    if is_high_volatility:
        # In crisis, exit faster. Any sign of weakness is a sell.
        if strong_reversal or trend_weakness or panic_news:
            sell_signal = True
    else: # Normal Mode
        # In normal markets, allow for more breathing room but still exit on clear weakness.
        if strong_reversal or panic_news or extreme_overbought:
            sell_signal = True
        # In a trending market, use trend_weakness as a primary exit signal.
        elif not (trend_strength < 0.005) and trend_weakness:
            sell_signal = True

    # --- Final Decision ---
    if buy_signal:
        return "BUY"
    elif sell_signal:
        return "SELL"
    else:
        return "HOLD"