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


def calculate_rsi(prices, period):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    
    try:
        import pandas as pd
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except (ImportError, ZeroDivisionError):
        # Fallback to numpy implementation
        prices_arr = np.array(prices, dtype=float)
        deltas = np.diff(prices_arr)
        if len(deltas) < period:
            return None
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use simple moving average for the first calculation
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Then use Wilder's smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0 # RSI is 100 if avg_loss is zero
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy with a dual-mode, regime-switching logic.
    - Crisis Mode (High Volatility): Prioritizes strong sentiment signals confirmed by trend.
    - Normal Mode (Low Volatility): Operates as a pure trend-following system, ignoring news noise.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Primarily for Crisis Mode) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Bullish
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5,
        # Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0,
        "downgrade": -1.5, "tariff": -1.5, "weak earnings": -2.0, "bankruptcy": -2.5,
        "negative outlook": -1.5, "contraction": -1.5
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
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOLATILITY_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from indicator calculations
    if short_ema is None or long_ema is None or rsi is None or np.isnan(rsi):
        return "HOLD"

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
    
    # --- 3. Adaptive Decision Logic based on Regime ---
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime (e.g., >2% daily std dev)

    if is_high_volatility:
        # --- CRISIS MODE ---
        # This logic was successful in past stress tests. It requires strong sentiment
        # confirmation for trades, which is effective in news-driven, volatile markets.
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
        # --- NORMAL MODE (SELF-IMPROVEMENT) ---
        # Past failures showed that sentiment is a noisy and ineffective filter in
        # low-volatility, trending markets. This mode is now a pure trend-following
        # system, using the EMA crossover as the primary signal, ignoring sentiment.
        # This prevents the strategy from being sidelined by minor negative news during an uptrend.
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30 # Retained for potential future use, but not in primary logic.

        bullish_trend_signal = short_ema > long_ema
        bearish_trend_signal = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT

        # BUY: Enter on a confirmed bullish trend, filtering for overbought conditions.
        if bullish_trend_signal and is_not_overbought:
            return "BUY"
        
        # SELL: Exit when the primary trend reverses. This is a pure technical signal,
        # making the exit faster and more reliable than waiting for a sentiment shift.
        elif bearish_trend_signal:
            return "SELL"
        
    # Default to HOLD if no clear signal is generated in either regime
    return "HOLD"