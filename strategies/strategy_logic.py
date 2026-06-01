import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
    ema_values[0] = np.mean(data_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(1, len(ema_values)):
        ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    macd_line_series = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line_series) < signal_period:
        return None, None
        
    signal_line_series = calculate_ema_series(macd_line_series, signal_period)
    
    if len(signal_line_series) == 0:
        return None, None
        
    macd_line = macd_line_series[-1]
    histogram = macd_line - signal_line_series[-1]
    return macd_line, histogram

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using a signal scoring system
    to increase adaptability and reduce passivity.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Expanded Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.5, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat expectations": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "persistent inflation": -2.5, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss expectations": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0
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
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100
    LONG_TERM_SMA_PERIOD = 200

    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1, LONG_TERM_SMA_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, macd_histogram = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    long_term_sma = calculate_sma(all_prices, LONG_TERM_SMA_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, macd_histogram, upper_band, long_term_sma]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Signal Scoring & Decision Logic ---
    buy_score = 0.0
    sell_score = 0.0

    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following scoring ===
        # Stricter thresholds, focus on strong, clear signals.
        BUY_THRESHOLD = 3.5
        SELL_THRESHOLD = 3.5

        # Trend Signal (Weight: 1.5)
        if short_ema > long_ema: buy_score += 1.5
        if short_ema < long_ema: sell_score += 1.5
        
        # Momentum Signal (Weight: 1.5)
        if macd_histogram > 0: buy_score += 1.5
        if macd_histogram < 0: sell_score += 1.5

        # Sentiment Signal (Weight: 1.0)
        if net_sentiment_score >= 2.5: buy_score += 1.0
        if net_sentiment_score <= -2.5: sell_score += 1.0
        
        # Overbought/Oversold Filter (Negative score to prevent chasing)
        if rsi > 70: buy_score -= 1.0
        if rsi < 30: sell_score -= 1.0

    else:
        # === UNIFIED NORMAL MODE: Flexible scoring for trend and mean-reversion ===
        BUY_THRESHOLD = 3.0
        SELL_THRESHOLD = 3.0

        # --- Trend-Following Signals ---
        # Trend Signal (Weight: 1.5)
        if short_ema > long_ema: buy_score += 1.5
        if short_ema < long_ema: sell_score += 1.5

        # Momentum Signal (Weight: 1.0)
        if macd_histogram > 0 and macd_line > 0: buy_score += 1.0
        if macd_histogram < 0 and macd_line < 0: sell_score += 1.0

        # Sentiment Signal (Weight: 1.0)
        if net_sentiment_score >= 1.0: buy_score += 1.0
        if net_sentiment_score <= -1.0: sell_score += 1.0

        # RSI Confirmation (Weight: 0.5)
        if 50 < rsi < 75: buy_score += 0.5
        if 25 < rsi < 50: sell_score += 0.5

        # --- Mean-Reversion Signals (with strong safety filter) ---
        # Buy the dip ONLY if the long-term trend is still up (Weight: 1.5)
        if rsi < 30 and current_price < lower_band and current_price > long_term_sma:
            buy_score += 1.5
        
        # Sell the rip (Weight: 1.5)
        if rsi > 70 and current_price > upper_band:
            sell_score += 1.5

    # --- 4. Final Decision ---
    if buy_score >= BUY_THRESHOLD and buy_score > sell_score:
        return "BUY"
    elif sell_score >= SELL_THRESHOLD and sell_score > buy_score:
        return "SELL"
    
    return "HOLD"