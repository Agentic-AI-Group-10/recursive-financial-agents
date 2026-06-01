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
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
        
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bbands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    # Calculate on the most recent 'period' days
    prices_slice = prices[-period:]
    sma = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return sma, upper_band, lower_band

def calculate_momentum(prices, period=5):
    """Calculates simple price momentum (Rate of Change)."""
    if len(prices) < period + 1:
        return None
    # Positive value indicates upward momentum, negative indicates downward.
    return prices[-1] - prices[-(period + 1)]

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses Bollinger Bands and
    a momentum filter to improve signal conviction.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0, "ai boom": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical tension": -2.0, "tightening": -1.5, "miss": -1.5, "downgrade": -1.5
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
    BBAND_PERIOD = 20
    MOMENTUM_PERIOD = 5
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1, BBAND_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bbands(all_prices, BBAND_PERIOD)
    momentum = calculate_momentum(all_prices, MOMENTUM_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, upper_band, momentum]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and rsi < RSI_OVERBOUGHT:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and rsi > RSI_OVERSOLD:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market with Momentum Confirmation
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            # Buy if trend is up, momentum is positive, sentiment is supportive
            if bullish_trend and momentum > 0 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
                return "BUY"
            # Sell if trend is down, momentum is negative, sentiment is supportive
            elif bearish_trend and momentum < 0 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion with Bollinger Bands)
            MEAN_REVERSION_RSI_OVERSOLD = 30
            MEAN_REVERSION_RSI_OVERBOUGHT = 70
            
            # Buy on extreme dips below the lower Bollinger Band, confirmed by RSI
            if current_price < lower_band and rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -1.5:
                return "BUY"
            # Sell on extreme rips above the upper Bollinger Band, confirmed by RSI
            elif current_price > upper_band and rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 1.5:
                return "SELL"

    return "HOLD"