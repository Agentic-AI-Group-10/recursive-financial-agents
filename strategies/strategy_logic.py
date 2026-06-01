import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # This loop is sufficient and clear for calculating the single final value.
    ema = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema = (prices_arr[i] - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method (equivalent to a specific EMA)
    # Initialize with a simple moving average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Smooth subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0 # RSI is 100 if avg_loss is zero
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD line, signal line, and histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None, None

    # Calculate Short and Long EMAs over the whole series to derive the MACD line
    short_ema = calculate_ema(prices, short_period)
    long_ema = calculate_ema(prices, long_period)
    
    if short_ema is None or long_ema is None:
        return None, None, None

    # To get the signal line, we need the history of the MACD line.
    # This requires calculating the full EMA series.
    prices_arr = np.array(prices, dtype=float)
    
    # Full EMA calculation for internal use
    def get_full_ema(data, period):
        ema_values = np.zeros_like(data)
        ema_values[period - 1] = np.mean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

    full_short_ema = get_full_ema(prices_arr, short_period)
    full_long_ema = get_full_ema(prices_arr, long_period)
    
    macd_line_series = full_short_ema[long_period-1:] - full_long_ema[long_period-1:]
    
    if len(macd_line_series) < signal_period:
        return None, None, None

    # Calculate the signal line (EMA of the MACD line)
    signal_line_ema = get_full_ema(macd_line_series, signal_period)
    
    macd_line_latest = macd_line_series[-1]
    signal_line_latest = signal_line_ema[-1]
    macd_histogram = macd_line_latest - signal_line_latest
    
    return macd_line_latest, signal_line_latest, macd_histogram


def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses MACD for trend/momentum
    confirmation and adapts its logic for volatile, trending, and choppy markets.

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
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 2.0, "disinflation": 2.0,
        "jobs report beats": 2.0, "cpi lower": 1.5, "guidance raised": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "hot inflation": -2.0, "cpi higher": -2.0, "sell-off": -2.0,
        "weak earnings": -2.0, "jobs report miss": -2.0, "guidance cut": -1.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "uncertainty": -1.0, "volatility": -1.0
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
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100
    MACD_SHORT = 12
    MACD_LONG = 26
    MACD_SIGNAL = 9

    # Ensure enough data for all indicators
    required_history_length = max(VOL_LONG_PERIOD + 1, MACD_LONG + MACD_SIGNAL)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line, macd_histogram = calculate_macd(all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line, macd_histogram]):
        return "HOLD"

    # Regime Detection: Volatility and Trend Strength (using MACD)
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    
    is_high_volatility = (short_term_vol > long_term_vol * 1.6) and (short_term_vol > 0.015)
    
    # Use normalized MACD histogram to detect choppy vs. trending markets
    normalized_macd_hist = abs(macd_histogram) / current_price
    is_choppy_market = normalized_macd_hist < 0.0015

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following with momentum filter ===
        # In volatile markets, only act on strong, confirmed signals.
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        # Buy on strong uptrend, positive sentiment, and accelerating momentum
        if short_ema > long_ema and macd_histogram > 0 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
            return "BUY"
        # Sell on strong downtrend, negative sentiment, and accelerating momentum
        elif short_ema < long_ema and macd_histogram < 0 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "SELL"
            
    elif is_choppy_market:
        # === CHOPPY MODE: Mean-Reversion Logic ===
        # In low-momentum, ranging markets, fade the extremes.
        MEAN_REVERSION_RSI_OVERSOLD = 28
        MEAN_REVERSION_RSI_OVERBOUGHT = 72
        
        # Buy on deep oversold conditions, if sentiment isn't catastrophic
        if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -2.0:
            return "BUY"
        # Sell on strong overbought conditions, if sentiment isn't euphoric
        elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 2.0:
            return "SELL"
            
    else:
        # === NORMAL TREND MODE: Standard trend-following with MACD confirmation ===
        BULLISH_SENTIMENT_THRESHOLD = 1.0
        BEARISH_SENTIMENT_THRESHOLD = -1.0
        RSI_OVERBOUGHT_FILTER = 75 # Avoid buying at extreme tops
        RSI_OVERSOLD_FILTER = 25   # Avoid selling at extreme bottoms

        # Buy on bullish EMA cross confirmed by positive MACD histogram and sentiment
        if short_ema > long_ema and macd_histogram > 0 and rsi < RSI_OVERBOUGHT_FILTER and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
            return "BUY"
        # Sell on bearish EMA cross confirmed by negative MACD histogram and sentiment
        elif short_ema < long_ema and macd_histogram < 0 and rsi > RSI_OVERSOLD_FILTER and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"