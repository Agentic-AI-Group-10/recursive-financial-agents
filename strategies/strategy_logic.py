import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a series of prices."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # Pandas implementation is more robust and standard
    try:
        import pandas as pd
        return pd.Series(prices_arr).ewm(span=period, adjust=False).mean().iloc[-1]
    except ImportError:
        # Fallback to numpy if pandas is not available
        ema_values = np.zeros_like(prices_arr, dtype=float)
        ema_values[0] = prices_arr[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(prices_arr)):
            ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[-1]


def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    # Use simple moving average for the initial calculation
    avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0.0
    avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0.0
    
    if avg_gain == 0 and avg_loss == 0: # Avoid division by zero if no changes
        return 50.0

    # Wilder's smoothing for subsequent values
    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            gain = delta
            loss = 0
        else:
            gain = 0
            loss = -delta
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD, Signal Line, and Histogram."""
    if len(prices) < long_period:
        return None, None, None
    
    short_ema = calculate_ema(prices, short_period)
    long_ema = calculate_ema(prices, long_period)
    
    if short_ema is None or long_ema is None:
        return None, None, None
        
    macd_line = short_ema - long_ema
    
    # To calculate the signal line, we need a history of MACD values
    # This is a proxy since we don't have the full history of MACD
    # We re-calculate it over the price history to get the signal line
    try:
        import pandas as pd
        prices_series = pd.Series(prices)
        ema_short = prices_series.ewm(span=short_period, adjust=False).mean()
        ema_long = prices_series.ewm(span=long_period, adjust=False).mean()
        macd_series = ema_short - ema_long
        signal_series = macd_series.ewm(span=signal_period, adjust=False).mean()
        
        macd_line_val = macd_series.iloc[-1]
        signal_line_val = signal_series.iloc[-1]
        histogram_val = macd_line_val - signal_line_val
        return macd_line_val, signal_line_val, histogram_val
    except ImportError:
        # Fallback without pandas is complex and less accurate for signal line
        return macd_line, None, None # Signal line calculation is non-trivial without history


def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy using a weighted scoring system based on MACD, RSI,
    and sentiment, with an adaptive volatility regime switch.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Enhanced Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Bullish (Economic Strength & Policy)
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat estimates": 2.0, "strong earnings": 2.0, "growth": 1.5,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "ai boom": 2.0, "innovation": 1.5,
        # Bearish (Economic Weakness & Policy)
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss estimates": -2.0, "weak earnings": -2.0, "inflation": -2.0,
        "sell-off": -2.0, "downgrade": -1.5, "tariff": -1.5, "bankruptcy": -2.5,
        "geopolitical tension": -2.0, "supply chain disruption": -1.5, "contraction": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Volatility Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    LONG_MACD_PERIOD = 26
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20

    # Ensure enough data for all indicators
    required_history_length = max(LONG_MACD_PERIOD, RSI_PERIOD + 1, VOLATILITY_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    macd_line, signal_line, macd_histogram = calculate_macd(all_prices)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from indicators
    if macd_line is None or signal_line is None or rsi is None:
        return "HOLD"

    # Calculate volatility using Coefficient of Variation for a normalized measure
    recent_prices = np.array(all_prices[-VOLATILITY_PERIOD:])
    volatility_cv = np.std(recent_prices) / np.mean(recent_prices)
    
    # --- 3. Adaptive Decision Logic based on Regime ---
    is_high_volatility = volatility_cv > 0.04  # Threshold for high-volatility (e.g., >4% CoV)

    if is_high_volatility:
        # Crisis Mode: Be more selective. Require higher conviction.
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        RSI_OVERBOUGHT = 68
        RSI_OVERSOLD = 32
        DECISION_THRESHOLD = 2.5 # Require at least two strong signals
    else:
        # Normal Mode: Be more active.
        BULLISH_SENTIMENT_THRESHOLD = 1.0
        BEARISH_SENTIMENT_THRESHOLD = -1.0
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30
        DECISION_THRESHOLD = 2.0 # Standard conviction level

    # --- 4. Weighted Scoring System ---
    buy_score = 0.0
    sell_score = 0.0

    # Signal 1: Trend Confirmation (MACD) - Weight: 1.5
    if macd_line > signal_line and macd_histogram > 0:
        buy_score += 1.5
    elif macd_line < signal_line and macd_histogram < 0:
        sell_score += 1.5

    # Signal 2: Sentiment Catalyst - Weight: 1.5
    if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
        buy_score += 1.5
    elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
        sell_score += 1.5

    # Signal 3: Momentum Filter/Contrarian (RSI) - Weight: 1.0
    # Penalize buying when overbought and selling when oversold.
    if rsi > RSI_OVERBOUGHT:
        sell_score += 1.0 # Contrarian signal: market is overbought, potential reversal
    if rsi < RSI_OVERSOLD:
        buy_score += 1.0 # Contrarian signal: market is oversold, potential bounce

    # --- 5. Final Decision ---
    # A BUY signal requires a score surpassing the threshold AND the trend not being bearish.
    if buy_score >= DECISION_THRESHOLD and macd_line > signal_line:
        return "BUY"
    
    # A SELL signal requires a score surpassing the threshold AND the trend not being bullish.
    elif sell_score >= DECISION_THRESHOLD and macd_line < signal_line:
        return "SELL"
        
    # Default to HOLD, reducing noise and preventing low-conviction trades.
    return "HOLD"