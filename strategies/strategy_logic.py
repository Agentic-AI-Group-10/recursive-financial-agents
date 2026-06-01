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

        # Use Wilder's smoothing for RSI
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
    SELF-IMPROVED: This version addresses the critical failure of underperforming
    in normal, trending markets by adapting its logic based on volatility.

    - In High-Volatility (Crisis) markets, it retains the successful contrarian
      logic that combines strong sentiment with RSI extremes.
    - In Normal-Volatility markets, it switches to a trend-following model,
      using the EMA crossover as the primary signal to avoid fighting the trend,
      a key lesson from past underperformance.
    """
    # --- 1. Sentiment Analysis (Unchanged) ---
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

    if short_ema is None or long_ema is None or rsi is None:
        return "HOLD"

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
    
    # --- 3. Adaptive Decision Logic based on Regime ---
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime

    bullish_trend = short_ema > long_ema
    bearish_trend = short_ema < long_ema

    # --- 4. Generate Trading Signals based on Regime ---
    if is_high_volatility:
        # CRISIS MODE: Retain the successful contrarian logic.
        # This mode excels in volatile, news-driven, mean-reverting markets.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65  # Tighter bands for quicker reactions
        RSI_OVERSOLD = 35

        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        # BUY: Strong positive news, bullish trend confirmation, and not overbought.
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        
        # SELL: Strong negative news, bearish trend confirmation, and not oversold.
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"
    
    else:
        # NORMAL MODE: NEW trend-following logic to address past failures.
        # This mode prioritizes riding sustained trends and is less sensitive to news noise
        # and non-extreme RSI levels.
        BULLISH_SENTIMENT_THRESHOLD = 0.5 # Require only mildly positive news to confirm trend
        BEARISH_SENTIMENT_THRESHOLD = -1.0
        RSI_EXTREME_OVERBOUGHT = 80 # Only stop buying if market is in a speculative frenzy
        RSI_OVERSOLD_EXIT_CONF = 40 # Use RSI to confirm exit, but don't let it block a trend-break sell

        # BUY: Primary signal is the bullish trend. Confirm with non-negative sentiment
        # and ensure RSI is not at an extreme speculative peak. This solves the problem
        # of RSI staying "overbought" (e.g., at 72) in a healthy uptrend.
        if bullish_trend and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and rsi < RSI_EXTREME_OVERBOUGHT:
            return "BUY"
        
        # SELL: Primary signal is the trend breaking down (bearish crossover).
        # Confirm with negative sentiment. The RSI check is removed for the primary exit
        # because if the trend is broken, we must exit.
        elif bearish_trend and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "SELL"
        
    # Default to HOLD if no high-conviction signal is found in the current regime
    return "HOLD"