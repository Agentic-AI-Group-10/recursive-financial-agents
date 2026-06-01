import numpy as np
import re

# --- Helper Functions for Indicators ---

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
        return 100.0 # RSI is 100 if there are no losses
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility(prices, short_period, long_period):
    """Calculates short-term and long-term historical volatility using log returns."""
    if len(prices) < long_period + 1:
        return None, None
    prices_arr = np.array(prices, dtype=float)
    # Use log returns for better statistical properties
    log_returns = np.log(prices_arr[1:] / prices_arr[:-1])
    if len(log_returns) < long_period:
        return None, None
    short_term_vol = np.std(log_returns[-short_period:])
    long_term_vol = np.std(log_returns[-long_period:])
    return short_term_vol, long_term_vol

def decide(current_price, price_history, news_context):
    """
    Self-improved adaptive trading strategy that adjusts its parameters based on market volatility.
    
    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.
        
    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    
    # --- 1. Data Preparation and Minimum Length Check ---
    all_prices = price_history + [current_price]
    
    # Define periods for indicators
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    SHORT_VOL_PERIOD = 20
    LONG_VOL_PERIOD = 60
    
    # Require enough data for all calculations, especially the long-term volatility
    required_history_length = LONG_VOL_PERIOD + 1
    if len(all_prices) < required_history_length:
        return "HOLD"

    # --- 2. Adaptive Regime Filter ---
    short_term_vol, long_term_vol = calculate_volatility(all_prices, SHORT_VOL_PERIOD, LONG_VOL_PERIOD)
    
    # Default to Normal Regime if volatility calculation fails
    is_crisis_regime = False
    if short_term_vol is not None and long_term_vol is not None and long_term_vol > 0:
        # A crisis is defined as short-term volatility being significantly higher than long-term
        if short_term_vol > long_term_vol * 1.5:
            is_crisis_regime = True

    # Set dynamic parameters based on the detected regime
    if is_crisis_regime:
        # In crisis, require higher conviction and be more cautious at extremes
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        RSI_OVERBOUGHT = 75
        RSI_OVERSOLD = 25
    else:
        # In normal markets, be more sensitive to signals to increase activity
        BULLISH_SENTIMENT_THRESHOLD = 1.0
        BEARISH_SENTIMENT_THRESHOLD = -1.0
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30

    # --- 3. Sentiment Analysis (Unchanged from Parent) ---
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

    # --- 4. Technical Indicator Calculation ---
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    
    if short_ema is None or long_ema is None or rsi is None:
        return "HOLD"

    # --- 5. Simplified & Robust Decision Logic ---
    bullish_tech_signal = False
    bearish_tech_signal = False
    
    # Bullish Signal: Trend is up (EMA cross), momentum is positive (price > short EMA), and not overbought.
    if short_ema > long_ema and current_price > short_ema and rsi < RSI_OVERBOUGHT:
        bullish_tech_signal = True
        
    # Bearish Signal: Trend is down (EMA cross), momentum is negative (price < short EMA), and not oversold.
    if short_ema < long_ema and current_price < short_ema and rsi > RSI_OVERSOLD:
        bearish_tech_signal = True

    # --- 6. Final Decision ---
    # BUY: Requires both bullish sentiment and a confirmed bullish technical signal.
    if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_tech_signal:
        return "BUY"
    
    # SELL: Requires both bearish sentiment and a confirmed bearish technical signal.
    elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_tech_signal:
        return "SELL"
    
    # HOLD: Default action if conditions are not met or are conflicting.
    else:
        return "HOLD"