import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # Using pandas for a more robust and standard EMA calculation
    try:
        import pandas as pd
        return pd.Series(prices_arr).ewm(span=period, adjust=False).mean().iloc[-1]
    except ImportError:
        # Fallback to numpy implementation if pandas is not available
        ema_values = np.zeros_like(prices_arr, dtype=float)
        ema_values[period - 1] = np.mean(prices_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices_arr)):
            ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[-1]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = deltas * (deltas > 0)
    losses = -deltas * (deltas < 0)

    # Using pandas for a robust rolling calculation
    try:
        import pandas as pd
        s_gains = pd.Series(gains)
        s_losses = pd.Series(losses)
        avg_gain = s_gains.ewm(com=period - 1, adjust=False).mean().iloc[-1]
        avg_loss = s_losses.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    except ImportError:
        # Fallback numpy implementation
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

def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy with an adaptive volatility regime and a new
    mean-reversion sub-mode for choppy markets.

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
        # Bullish
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat estimates": 2.0, "surge": 2.0, "strong growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5, "fed pivot": 3.0,
        # Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss estimates": -2.0, "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0,
        "downgrade": -1.5, "tariff": -1.5, "weak earnings": -2.0, "bankruptcy": -2.5,
        "negative outlook": -1.5, "geopolitical risk": -2.0, "supply chain": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Volatility Regime ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20
    VOLATILITY_HISTORY = 252 # Look back one trading year for percentile

    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOLATILITY_PERIOD + VOLATILITY_HISTORY)
    if len(all_prices) < max(LONG_EMA_PERIOD + 1, VOLATILITY_PERIOD + 1): # Basic check
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    if short_ema is None or long_ema is None or rsi is None:
        return "HOLD"

    # Adaptive Volatility Regime Detection
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    if len(log_returns) < VOLATILITY_PERIOD:
        return "HOLD"

    # Calculate rolling volatility to find a dynamic threshold
    lookback_len = min(len(log_returns) - VOLATILITY_PERIOD + 1, VOLATILITY_HISTORY)
    rolling_vols = [np.std(log_returns[i:i+VOLATILITY_PERIOD]) for i in range(len(log_returns) - VOLATILITY_PERIOD + 1)]
    
    current_volatility = rolling_vols[-1]
    # Use 75th percentile of recent volatility as the high-volatility threshold
    volatility_threshold = np.percentile(rolling_vols[-lookback_len:], 75) if len(rolling_vols) > 1 else 0.02
    
    is_high_volatility = current_volatility > volatility_threshold

    # --- 3. Adaptive Decision Logic based on Regime ---
    if is_high_volatility:
        # Crisis Mode: High-conviction trend-following. Stricter thresholds.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"

    else:
        # Normal Mode: Hybrid Trend-Following and Mean-Reversion
        ema_spread_normalized = abs(short_ema - long_ema) / current_price
        is_choppy_market = ema_spread_normalized < 0.005 # Less than 0.5% spread indicates chop

        if is_choppy_market:
            # Sub-Regime: Mean Reversion for choppy markets
            # Buy on deep oversold, sell on deep overbought, if sentiment isn't strongly opposing
            if rsi < 25 and net_sentiment_score > -1.0:
                return "BUY"
            elif rsi > 75 and net_sentiment_score < 1.0:
                return "SELL"
        else:
            # Sub-Regime: Trend-Following for trending normal markets
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30

            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            is_not_overbought = rsi < RSI_OVERBOUGHT
            is_not_oversold = rsi > RSI_OVERSOLD

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
                return "SELL"

    # Default to HOLD if no high-conviction signal is found
    return "HOLD"