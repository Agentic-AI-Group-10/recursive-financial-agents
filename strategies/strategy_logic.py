import numpy as np
import pandas as pd
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a list of prices."""
    if len(prices) < period:
        return [None] * len(prices)
    return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    
    series = pd.Series(prices)
    delta = series.diff(1)
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
    
    # Use Wilder's smoothing for subsequent values
    # For a single calculation, we can simulate the last step
    # Note: A full historical calculation would be slightly different, but this is a robust approximation for the final value.
    # For simplicity and performance, we'll use the rolling mean which is a common and valid approach.

    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(prices, high_prices, low_prices, period=14):
    """Calculates the Average True Range (ATR) for the latest day."""
    if len(prices) < period or len(high_prices) < period or len(low_prices) < period:
        return None
    
    # For this function, we assume high/low are not available, so we approximate.
    # A robust approximation is to use the daily change.
    # In a real system, high/low would be passed in.
    # Here, we'll use `abs(price_t - price_t-1)` as a proxy for True Range.
    tr = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    
    if not tr:
        return None
        
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]
    return atr

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD line, Signal line, and Histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None
    
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    
    if ema_short[-1] is None or ema_long[-1] is None:
        return None, None
        
    macd_line_full = [s - l if s is not None and l is not None else 0 for s, l in zip(ema_short, ema_long)]
    
    signal_line_full = calculate_ema(macd_line_full, signal_period)

    if signal_line_full[-1] is None:
        return None, None

    return macd_line_full[-1], signal_line_full[-1]

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses MACD for confirmation
    and ATR for dynamic volatility and trend-strength assessment.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Expanded Economic Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.5,
        "dovish": 2.0, "record high": 1.5, "bullish": 2.0, "surge": 1.5,
        "strong earnings": 2.0, "cooling inflation": 2.0, "disinflation": 2.0,
        "cpi lower": 2.0, "jobs report strong": 1.5, "beat estimates": 1.5,
        "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -3.0, "crisis": -3.0, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation fears": -2.5, "sell-off": -2.0, "weak earnings": -2.0,
        "cpi higher": -2.5, "jobs report weak": -2.0, "miss estimates": -1.5,
        "tightening": -1.5, "downgrade": -1.5, "tariff": -1.5, "vix spike": -2.0,
        "geopolitical risk": -2.0
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
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    ATR_SHORT_PERIOD = 10
    ATR_LONG_PERIOD = 50

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + SIGNAL_PERIOD, ATR_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)[-1]
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)[-1]
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)
    
    # Use a simplified ATR since high/low are not available
    atr_short = calculate_atr(all_prices, all_prices, all_prices, ATR_SHORT_PERIOD)
    atr_long = calculate_atr(all_prices, all_prices, all_prices, ATR_LONG_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line, atr_short, atr_long]):
        return "HOLD"

    # ATR-based Volatility Regime: More robust than std dev
    is_high_volatility = atr_short > (atr_long * 1.75)

    # --- 3. Multi-Regime Decision Logic with MACD Confirmation ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, momentum-confirmed trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT = 68
        RSI_OVERSOLD = 32

        is_bullish_trend = short_ema > long_ema and macd_line > signal_line
        is_bearish_trend = short_ema < long_ema and macd_line < signal_line
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < RSI_OVERBOUGHT:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > RSI_OVERSOLD:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        # Use ATR to dynamically define a choppy market
        trend_strength = abs(short_ema - long_ema)
        is_choppy_market = trend_strength < (atr_long * 0.3) # Trend is weak if EMA diff < 30% of long-term ATR

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market with MACD confirmation
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            is_bullish_trend = short_ema > long_ema and macd_line > signal_line
            is_bearish_trend = short_ema < long_ema and macd_line < signal_line

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < RSI_OVERBOUGHT:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > RSI_OVERSOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            # Buy on extreme oversold conditions, provided sentiment isn't catastrophic.
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -2.0:
                return "BUY"
            # Sell on extreme overbought conditions, provided sentiment isn't euphoric.
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 2.0:
                return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"