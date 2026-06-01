import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # Using a simplified calculation for the final value which is mathematically equivalent
    # and avoids creating a large intermediate array.
    ema = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for price in prices_arr[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

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
    A self-improved, multi-regime trading strategy with a fast-acting panic-selling
    mechanism to protect against rapid market crashes.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Panic Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -3.0, "bankruptcy": -3.0,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        # Panic/Systemic Risk Keywords
        "contagion": -4.0, "liquidity crisis": -4.0, "credit crunch": -4.0, "panic": -3.5
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
    FAST_EMA_PERIOD = 10   # For fast momentum checks
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    fast_ema = calculate_ema(all_prices, FAST_EMA_PERIOD)
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from calculations
    if fast_ema is None or short_ema is None or long_ema is None or rsi is None:
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. NEW: Panic Sell / Waterfall Decline Protection ---
    # This is a fast-acting circuit breaker to exit before slow indicators confirm a crash.
    # It triggers if the price drops sharply below its short-term trend anchor (fast EMA).
    is_in_freefall = current_price < (fast_ema * 0.97) # Price is >3% below the 10-day EMA
    if is_in_freefall:
        return "SELL" # Override all other logic for capital preservation

    # --- 4. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: Symmetrical Trend-Following with Stricter Entry ---
        # In volatile markets, follow the dominant trend but be highly selective.
        # SELL logic is now as aggressive as BUY logic to handle sustained downtrends.
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        # BUY: Positive news + price must reclaim fast EMA to prove bounce has strength (avoids bull traps)
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and current_price > fast_ema and short_ema > long_ema:
            return "BUY"
        # SELL: Negative news + price breaks below fast EMA, indicating loss of momentum
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and current_price < fast_ema and short_ema < long_ema:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ---
        # This logic performed well and is kept largely intact.
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and rsi < RSI_OVERBOUGHT:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and rsi > RSI_OVERSOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -2.0:
                return "BUY"
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 2.0:
                return "SELL"

    return "HOLD"