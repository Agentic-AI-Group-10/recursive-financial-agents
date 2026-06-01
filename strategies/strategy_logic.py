import numpy as np
import re
from collections import deque

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a series of prices."""
    if len(prices) < period:
        return None
    # Using deque for potential efficiency, but numpy array is fine
    prices_arr = np.array(prices, dtype=float)
    # The first EMA is a simple moving average
    ema_values = [np.mean(prices_arr[:period])]
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        new_ema = (prices_arr[i] - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(new_ema)
    return np.array(ema_values)

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method which is common for RSI
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

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD line and Signal line for a series of prices."""
    if len(prices) < long_period:
        return None, None
    
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    
    # Align arrays by taking the tail of the shorter one
    macd_line = ema_short[len(ema_short) - len(ema_long):] - ema_long
    
    if len(macd_line) < signal_period:
        return None, None
        
    signal_line = calculate_ema(macd_line, signal_period)
    
    # Return aligned MACD and Signal lines
    return macd_line[len(macd_line) - len(signal_line):], signal_line


def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy that uses an adaptive volatility regime to switch
    between a high-conviction trend-following model (for crises) and a more sensitive
    momentum-based model (for normal markets).

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
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5,
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

    # --- 2. Technical Indicators & Adaptive Volatility Regime ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20
    VOLATILITY_LOOKBACK = 100 # For adaptive threshold

    # Ensure enough data for all indicators
    required_history_length = LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD + 2
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    
    # Safeguard against None values
    if rsi is None or macd_line is None or len(macd_line) < 2:
        return "HOLD"

    # Adaptive Volatility Regime Detection
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    if len(log_returns) < VOLATILITY_PERIOD + VOLATILITY_LOOKBACK:
        # Not enough data for adaptive threshold, use static fallback
        current_volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
        is_high_volatility = current_volatility > 0.02
    else:
        # Calculate rolling volatility to create an adaptive threshold
        rolling_vol = [np.std(log_returns[i:i+VOLATILITY_PERIOD]) for i in range(len(log_returns) - VOLATILITY_PERIOD + 1)]
        long_term_avg_vol = np.mean(rolling_vol[-VOLATILITY_LOOKBACK:-1]) # Avg vol over lookback period
        current_volatility = rolling_vol[-1]
        # High volatility is when current vol is 1.5x its recent average
        is_high_volatility = current_volatility > (long_term_avg_vol * 1.5)

    # --- 3. Adaptive Decision Logic based on Regime ---
    if is_high_volatility:
        # CRISIS MODE: Use the proven high-conviction logic from the parent strategy.
        # Requires clear trend (EMA crossover), strong sentiment, and RSI confirmation.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        # Calculate EMAs only if needed for this regime
        ema_short_full = calculate_ema(all_prices, SHORT_EMA_PERIOD)
        ema_long_full = calculate_ema(all_prices, LONG_EMA_PERIOD)
        if ema_short_full is None or ema_long_full is None:
            return "HOLD"
        
        short_ema = ema_short_full[-1]
        long_ema = ema_long_full[-1]

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if (net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and 
            bullish_trend and 
            rsi < RSI_OVERBOUGHT):
            return "BUY"
        
        elif (net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and 
              bearish_trend and 
              rsi > RSI_OVERSOLD):
            return "SELL"

    else:
        # NORMAL MODE: Use a more sensitive momentum-based logic (MACD Crossover)
        # to address the "late entry" and "overly restrictive" issues.
        BULLISH_SENTIMENT_THRESHOLD = 0.5 # Lower threshold for sensitivity
        BEARISH_SENTIMENT_THRESHOLD = -0.5
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30

        # Check for MACD Crossover
        bullish_momentum_signal = macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]
        bearish_momentum_signal = macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]
        
        if (bullish_momentum_signal and 
            net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and 
            rsi < RSI_OVERBOUGHT):
            return "BUY"
            
        elif (bearish_momentum_signal and 
              net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and 
              rsi > RSI_OVERSOLD):
            return "SELL"

    # Default to HOLD if no strong signal is found in the active regime
    return "HOLD"