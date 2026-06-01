import numpy as np
import re
import pandas as pd # Using pandas for more robust EMA calculation

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """
    Calculates the Exponential Moving Average (EMA) for the latest price using pandas for robustness.
    """
    if len(prices) < period:
        return None
    # Using pandas is a standard and numerically stable way to calculate EMA
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_rsi(prices, period):
    """
    Calculates the Relative Strength Index (RSI) for the latest price.
    """
    if len(prices) < period + 1:
        return None
    
    series = pd.Series(prices)
    delta = series.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Using Wilder's smoothing (alpha = 1/period) which is standard for RSI
    # This is equivalent to the loop in the parent but more concise.
    # For the first value, it's a simple average.
    # For subsequent values, it's smoothed.
    delta = delta.iloc[1:]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    if avg_loss.iloc[-1] == 0:
        return 100.0
    
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    rsi = 100 - (100 / (1 + rs))
    return rsi

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that incorporates a fast-acting
    "circuit breaker" to defend against high-velocity crashes, addressing a key
    failure mode from past runs.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Unchanged from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
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
    ULTRA_SHORT_EMA_PERIOD = 5 # New: For price stabilization check in crisis
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
    ultra_short_ema = calculate_ema(all_prices, ULTRA_SHORT_EMA_PERIOD)
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [ultra_short_ema, short_ema, long_ema, rsi]):
        return "HOLD"

    # Adaptive Volatility Regime: Compare short-term vs long-term volatility
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---

    # === NEW: Capital Preservation Circuit Breaker ===
    # Addresses the primary failure mode from the COVID crash analysis.
    # If volatility spikes and the price breaks below a key long-term average,
    # SELL IMMEDIATELY. This is a much faster signal than an EMA crossover.
    if is_high_volatility and current_price < long_ema:
        return "SELL"

    if is_high_volatility:
        # === CRISIS MODE: Enhanced with stricter BUY conditions ===
        # The circuit breaker handles the primary defense. This logic now focuses on
        # finding extremely high-conviction rebound entries, avoiding bull traps.
        BULLISH_SENTIMENT_THRESHOLD = 2.5 # Increased threshold for higher conviction
        RSI_OVERBOUGHT = 65
        
        bullish_trend_confirmed = short_ema > long_ema
        
        # IMPROVEMENT: To avoid buying a falling knife on "stimulus" news,
        # we now require price to show signs of stabilization by crossing
        # above a very short-term EMA.
        price_stabilized = current_price > ultra_short_ema

        if (net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and
            bullish_trend_confirmed and
            price_stabilized and
            rsi < RSI_OVERBOUGHT):
            return "BUY"
    else:
        # === NORMAL MODE: Adaptive (Unchanged from successful parent) ===
        # This logic performed well in normal market conditions.
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            if (net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and
                short_ema > long_ema and
                rsi < RSI_OVERBOUGHT):
                return "BUY"
            elif (net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and
                  short_ema < long_ema and
                  rsi > RSI_OVERSOLD):
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -1.5:
                return "BUY"
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 1.5:
                return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"