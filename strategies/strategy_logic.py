import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(prices, period):
    """Calculates a series of Exponential Moving Averages (EMAs)."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.full(len(prices_arr), np.nan)
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

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

    # Use Wilder's smoothing method
    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)
    avg_gain[period-1] = np.mean(gains[:period])
    avg_loss[period-1] = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
    final_avg_loss = avg_loss[-1]
    if final_avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain[-1] / final_avg_loss
        
    rsi = 100 - (100 / (1 + rs))
    return rsi

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that adds a momentum filter
    to avoid reversals and enhances choppy market detection to reduce whipsaws.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Enhanced Sentiment Analysis with Nuanced Phrases ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish Phrases
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus package": 2.5, "soft landing": 2.0,
        "cooling inflation": 2.0, "inflation peaking": 2.0, "strong jobs report": 2.0,
        "dovish": 2.0, "strong earnings": 2.0, "beats estimates": 1.5, "economic growth": 1.5,
        # High-Impact Bearish Phrases
        "rate hike": -2.5, "recession fears": -2.5, "hard landing": -2.5, "stagflation": -2.5,
        "persistent inflation": -2.5, "consumer confidence falls": -2.0, "weak jobs report": -2.0,
        "hawkish": -2.0, "weak earnings": -2.0, "misses estimates": -1.5, "geopolitical risk": -1.5,
        # General Keywords (lower weight)
        "bullish": 1.5, "surge": 1.5, "recovery": 1.0, "upgrade": 1.0,
        "bearish": -1.5, "plunge": -1.5, "sell-off": -1.5, "downgrade": -1.0, "inflation": -1.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators, Regime Detection & Momentum Filter ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100
    MOMENTUM_PERIOD = 5 # For EMA slope calculation

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate indicator series
    short_ema_series = calculate_ema_series(all_prices, SHORT_EMA_PERIOD)
    long_ema_series = calculate_ema_series(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from calculations
    if short_ema_series is None or long_ema_series is None or rsi is None:
        return "HOLD"
    
    short_ema = short_ema_series[-1]
    long_ema = long_ema_series[-1]

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # NEW: Momentum Filter (EMA Slope) to avoid entering stalling trends
    if len(short_ema_series) >= MOMENTUM_PERIOD:
        # Positive slope indicates upward momentum, negative indicates downward
        ema_momentum = short_ema - short_ema_series[-MOMENTUM_PERIOD]
    else:
        ema_momentum = 0

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following with momentum confirmation ===
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if bullish_trend and ema_momentum > 0 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and rsi < RSI_OVERBOUGHT:
            return "BUY"
        elif bearish_trend and ema_momentum < 0 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and rsi > RSI_OVERSOLD:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive logic with improved choppy market detection ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        
        # NEW: More robust choppy market detection
        recent_prices = all_prices[-VOL_SHORT_PERIOD:]
        price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
        is_choppy_market = trend_strength < 0.005 or price_range < 0.04

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market with momentum confirmation
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            if bullish_trend and ema_momentum > 0 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and rsi < RSI_OVERBOUGHT:
                return "BUY"
            elif bearish_trend and ema_momentum < 0 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and rsi > RSI_OVERSOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -1.5:
                return "BUY"
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 1.5:
                return "SELL"

    return "HOLD"