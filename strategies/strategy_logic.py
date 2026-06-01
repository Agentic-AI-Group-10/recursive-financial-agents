import numpy as np
import re
from collections import deque

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a series of prices."""
    if len(prices) < period:
        return np.array([None] * len(prices)) # Return array of Nones if not enough data
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.zeros_like(prices_arr, dtype=float)
    ema_values[:period-1] = np.nan # Set initial values to NaN
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use a deque for efficient rolling window calculation
    gain_window = deque(gains[:period])
    loss_window = deque(losses[:period])
    
    avg_gain = sum(gain_window) / period
    avg_loss = sum(loss_window) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD line, Signal line, and Histogram."""
    if len(prices) < long_period + signal_period:
        return None, None

    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    
    macd_line = ema_short - ema_long
    
    # Remove NaNs before calculating signal line EMA
    macd_line_valid = macd_line[~np.isnan(macd_line)]
    if len(macd_line_valid) < signal_period:
        return None, None

    signal_line = calculate_ema(macd_line_valid, signal_period)
    
    return macd_line[-1], signal_line[-1]


def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy that retains the successful "Crisis Mode" for high
    volatility while implementing a more responsive MACD-based system for "Normal Mode"
    to address past underperformance and reduce drawdowns.

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
        # Bullish
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5, "fed pivot": 2.5,
        # Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0,
        "downgrade": -1.5, "tariff": -1.5, "weak earnings": -2.0, "bankruptcy": -2.5,
        "negative outlook": -1.5, "contraction": -1.5, "geopolitical tension": -2.0
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
    SIGNAL_PERIOD = 9
    TREND_FILTER_PERIOD = 50 # New: Long-term trend filter
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20

    # Ensure enough data for all indicators
    required_history_length = TREND_FILTER_PERIOD + SIGNAL_PERIOD 
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)
    long_term_trend_ema = calculate_ema(all_prices, TREND_FILTER_PERIOD)[-1]

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[-VOLATILITY_PERIOD:] / np.array(all_prices)[-VOLATILITY_PERIOD-1:-1])
    volatility = np.std(log_returns)
    
    # Safeguard against None values from indicator calculations
    if rsi is None or macd_line is None or signal_line is None or long_term_trend_ema is None:
        return "HOLD"

    # --- 3. Adaptive Decision Logic based on Regime ---
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime

    if is_high_volatility:
        # --- CRISIS MODE (Retain successful parent logic) ---
        # Be more selective. Use stricter thresholds and proven EMA crossover.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)[-1]
        long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)[-1]
        if short_ema is None or long_ema is None:
            return "HOLD"

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"
            
    else:
        # --- NORMAL MODE (New, improved logic to address underperformance) ---
        # Use a more responsive MACD signal filtered by a long-term trend EMA.
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30
        
        # Define trend direction using the 50-period EMA
        is_uptrend = current_price > long_term_trend_ema
        is_downtrend = current_price < long_term_trend_ema

        # Define MACD signals
        is_macd_bullish_cross = macd_line > signal_line
        is_macd_bearish_cross = macd_line < signal_line

        # Define RSI conditions
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        # Define sentiment guardrails (less strict than crisis mode)
        sentiment_permits_buy = net_sentiment_score > -1.0
        sentiment_permits_sell = net_sentiment_score < 1.0

        # BUY Signal: Must be in a long-term uptrend with a fresh bullish MACD cross.
        # RSI and sentiment act as final checks to avoid bad entries.
        if is_uptrend and is_macd_bullish_cross and is_not_overbought and sentiment_permits_buy:
            return "BUY"
        
        # SELL Signal: Must be in a long-term downtrend with a fresh bearish MACD cross.
        # This provides a much faster exit than the old EMA crossover, reducing drawdowns.
        elif is_downtrend and is_macd_bearish_cross and is_not_oversold and sentiment_permits_sell:
            return "SELL"

    # Default to HOLD if no high-conviction signal is found
    return "HOLD"