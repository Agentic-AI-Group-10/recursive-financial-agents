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
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_efficiency_ratio(prices, period):
    """
    Calculates the Efficiency Ratio (ER) to measure trend strength.
    ER close to 1 indicates a strong trend; ER close to 0 indicates a choppy market.
    """
    if len(prices) < period + 1:
        return None
    
    prices_arr = np.array(prices[-period-1:], dtype=float)
    direction = abs(prices_arr[-1] - prices_arr[0])
    volatility = np.sum(np.abs(np.diff(prices_arr)))
    
    if volatility == 0:
        return 1.0 # If no price change, it's perfectly efficient (though not trending)
    
    return direction / volatility

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that adapts its core logic
    based on volatility, trend strength (via Efficiency Ratio), and momentum confirmation.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Enhanced Sentiment Analysis with Co-occurrence Boosters ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "ai boom": 2.0,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "geopolitical tension": -2.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    found_keywords = set()
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, context_lower):
            # Simplified check for negation for performance
            pre_context = context_lower.split(keyword, 1)[0][-30:]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            if not is_negated:
                net_sentiment_score += weight
                found_keywords.add(keyword)

    # Co-occurrence boosters
    if "rate cut" in found_keywords and "dovish" in found_keywords: net_sentiment_score += 1.0
    if "strong earnings" in found_keywords and "beat" in found_keywords: net_sentiment_score += 1.0
    if "recession" in found_keywords and "inflation" in found_keywords: net_sentiment_score -= 1.5
    if "rate hike" in found_keywords and "hawkish" in found_keywords: net_sentiment_score -= 1.0

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100
    ER_PERIOD = 20 # Efficiency Ratio period
    MOMENTUM_CONFIRM_PERIOD = 10 # Period for price high/low confirmation

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1, ER_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    efficiency_ratio = calculate_efficiency_ratio(all_prices, ER_PERIOD)

    if short_ema is None or long_ema is None or rsi is None or efficiency_ratio is None:
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.6) and (long_term_vol > 0.005)

    # --- 3. Multi-Regime Decision Logic with Momentum Confirmation ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following with momentum confirmation ===
        bullish_trend = short_ema > long_ema
        is_recent_high = current_price >= max(all_prices[-MOMENTUM_CONFIRM_PERIOD:])
        
        if bullish_trend and is_recent_high and net_sentiment_score >= 1.5 and rsi < 70:
            return "BUY"
        
        bearish_trend = short_ema < long_ema
        is_recent_low = current_price <= min(all_prices[-MOMENTUM_CONFIRM_PERIOD:])

        if bearish_trend and is_recent_low and net_sentiment_score <= -1.5 and rsi > 30:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive based on trend strength (Efficiency Ratio) ===
        is_choppy_market = efficiency_ratio < 0.3 # Low ER indicates a choppy/ranging market

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            bullish_trend = short_ema > long_ema
            is_recent_high = current_price >= max(all_prices[-MOMENTUM_CONFIRM_PERIOD:])
            
            if bullish_trend and is_recent_high and net_sentiment_score >= 1.0 and rsi < 75:
                return "BUY"

            bearish_trend = short_ema < long_ema
            is_recent_low = current_price <= min(all_prices[-MOMENTUM_CONFIRM_PERIOD:])

            if bearish_trend and is_recent_low and net_sentiment_score <= -1.0 and rsi > 25:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            # Buy deep dips and sell strong rips, as trends are unreliable.
            if rsi < 25 and net_sentiment_score > -2.0:
                return "BUY"
            elif rsi > 75 and net_sentiment_score < 2.0:
                return "SELL"

    return "HOLD"