import numpy as np
import re

# --- Helper Functions for Technical Indicators (Unchanged) ---
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

def calculate_rsi(prices, period):
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
    Self-improved trading strategy using a flexible scoring system to address
    the issue of low trading frequency from overly restrictive conditions.
    
    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.
        
    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    
    # --- 1. Sentiment Analysis (Unchanged - Proven Robustness) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "gdp rise": 2.0, 
        "rate cut": 2.5, "upgrade": 1.5, "stimulus": 2.0, "recovery": 1.5,
        "strong earnings": 2.0, "positive outlook": 1.5, "expansion": 1.5,
        "optimistic": 1.0, "record high": 2.0, "breakout": 1.5, "acquisition": 1.0,

        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "recession": -2.5, 
        "rate hike": -2.5, "inflation": -2.0, "downgrade": -1.5, "crisis": -2.5, 
        "tariff": -1.5, "weak earnings": -2.0, "negative outlook": -1.5, "contraction": -1.5,
        "pessimistic": -1.0, "sell-off": -2.0, "decline": -1.5, "bankruptcy": -2.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "decline in", "without", "struggle to"]
    
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicator Calculation ---
    all_prices = price_history + [current_price]
    
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    
    if short_ema is None or long_ema is None or rsi is None:
        return "HOLD" 

    # --- 3. Flexible Scoring-Based Decision Logic ---
    # This new logic addresses the core failure of being too restrictive by
    # allowing different combinations of signals to trigger a trade.
    
    bullish_score = 0.0
    bearish_score = 0.0
    
    # Condition 1: Trend Confirmation (EMA Crossover) - Weight: 1.5
    if short_ema > long_ema:
        bullish_score += 1.5
    elif short_ema < long_ema:
        bearish_score += 1.5
        
    # Condition 2: Price Action / Momentum (Price vs. Short EMA) - Weight: 1.0
    if current_price > short_ema:
        bullish_score += 1.0
    elif current_price < short_ema:
        bearish_score += 1.0
        
    # Condition 3: RSI Momentum - Weight: 0.5
    if rsi > 55: # Bullish momentum
        bullish_score += 0.5
    elif rsi < 45: # Bearish momentum
        bearish_score += 0.5
        
    # Condition 4: Sentiment Confirmation - Weight: 1.0
    if net_sentiment_score >= 1.0:
        bullish_score += 1.0
    elif net_sentiment_score <= -1.0:
        bearish_score += 1.0

    # --- 4. Final Decision with Risk Management Filters ---
    
    # Relaxed RSI thresholds to capture stronger trends, addressing past lessons.
    RSI_OVERBOUGHT = 75
    RSI_OVERSOLD = 25
    SCORE_THRESHOLD = 2.5 # Requires at least two strong signals to align

    # BUY Condition: Strong bullish alignment, but not in extreme overbought territory.
    if bullish_score >= SCORE_THRESHOLD and rsi < RSI_OVERBOUGHT:
        return "BUY"
    
    # SELL Condition: Strong bearish alignment, but not in extreme oversold territory.
    elif bearish_score >= SCORE_THRESHOLD and rsi > RSI_OVERSOLD:
        return "SELL"
    
    # Default to HOLD if conviction is not high enough.
    else:
        return "HOLD"