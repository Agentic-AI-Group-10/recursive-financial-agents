import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators (Self-Improved with EMA Slope) ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta >= 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    short_ema_full_series = calculate_ema_series(prices, short_period)
    long_ema_full_series = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter period's EMA
    macd_line = short_ema_full_series[long_period-short_period:] - long_ema_full_series
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_ema_slope(ema_series, period=5):
    """Calculates the slope of an EMA series over a short period."""
    if len(ema_series) < period:
        return None
    # Simple slope: (y2 - y1) / (x2 - x1). Here x2-x1 is the period.
    # We normalize by the last price to make it scale-invariant.
    return (ema_series[-1] - ema_series[-period]) / (ema_series[-1] * 0.01) if ema_series[-1] != 0 else 0

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy using a Signal Strength scoring system to aggregate
    evidence from multiple technical and sentiment factors, reducing false signals.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Refined with dampeners and new keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "capitulation": 2.0,
        # Moderate Positive
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "disinflation": 2.0, "market rally": 2.0, "vix crush": 2.0,
        # Mild Positive
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "easing tensions": 1.5, "consumer confidence": 1.5, "weak jobs report": 2.0, "de-escalation": 2.0,
        # Strong Negative
        "recession": -3.5, "crisis": -3.5, "stagflation": -3.5, "hot inflation": -3.0, "sticky inflation": -3.0,
        "war": -3.0, "conflict": -3.0, "yield curve inversion": -4.0, "quantitative tightening": -3.0,
        # Moderate Negative
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "sanctions": -2.5, "credit crunch": -2.5, "cpi beat": -2.5, "euphoria": -2.0, "vix spike": -2.5,
        # Mild Negative
        "hawkish": -2.0, "bearish": -2.0, "plunge": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "tightening": -1.5, "miss estimates": -1.5,
        "downgrade": -1.5, "tariff": -1.5, "uncertainty": -1.5, "strong jobs report": -2.0, "supply chain disruption": -2.0,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    dampener_words = ["may", "could", "potentially", "expects", "forecasts", "might"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            is_dampened = any(damp_word in pre_context for damp_word in dampener_words)
            
            current_weight = weight
            if is_negated:
                current_weight = -weight
            if is_dampened:
                current_weight *= 0.6
            net_sentiment_score += current_weight

    # --- 2. Technical Indicators & Regime Detection ---
    all_prices = price_history + [current_price]
    
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 50 # Lengthened for more stable trend identification
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50
    
    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema_series = calculate_ema_series(all_prices, SHORT_EMA_PERIOD)
    long_ema_series = calculate_ema_series(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices, short_period=SHORT_EMA_PERIOD, long_period=LONG_EMA_PERIOD)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    long_ema_slope = calculate_ema_slope(long_ema_series, period=5)

    if any(v is None for v in [rsi, upper_band, lower_band, short_atr, long_atr, long_ema_slope]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    short_ema = short_ema_series[-1]
    long_ema = long_ema_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Regime Detection
    is_high_volatility = short_atr > (long_atr * 1.5)
    is_trending = abs(long_ema_slope) > 0.15 # Trend is significant if slope is steep

    # --- 3. Signal Strength Scoring System ---
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend Signals (Primary driver)
    if short_ema > long_ema:
        bullish_score += 1.5
    else:
        bearish_score += 1.5
    if long_ema_slope > 0.05: # Positive slope
        bullish_score += 2.0 * min(long_ema_slope, 3.0) # Cap the contribution
    elif long_ema_slope < -0.05: # Negative slope
        bearish_score += 2.0 * min(abs(long_ema_slope), 3.0)

    # Momentum Signals
    if macd_histogram > 0:
        bullish_score += 1.0
    else:
        bearish_score += 1.0
    if macd_histogram > prev_macd_histogram: # Momentum increasing
        bullish_score += 0.5
    else: # Momentum decreasing
        bearish_score += 0.5
    
    # Overbought/Oversold Signals (Mean Reversion)
    rsi_overbought = 75 if is_high_volatility else 70
    rsi_oversold = 25 if is_high_volatility else 30
    
    if rsi < rsi_oversold:
        bullish_score += 1.5 # Potential reversal buy signal
    if rsi > rsi_overbought:
        bearish_score += 1.5 # Potential reversal sell signal

    if current_price < lower_band:
        bullish_score += 1.0 # Mean reversion signal
    if current_price > upper_band:
        bearish_score += 1.0 # Mean reversion signal

    # Sentiment Overlay
    if net_sentiment_score > 0:
        bullish_score += net_sentiment_score
    else:
        bearish_score += abs(net_sentiment_score)

    # --- 4. Decision Logic based on Scores & Regime ---
    BUY_THRESHOLD = 5.0
    SELL_THRESHOLD = 5.0
    
    if is_trending:
        # In a trend, prioritize trend-following signals and ignore mean-reversion signals
        if long_ema_slope > 0: # Bullish trend
            if bullish_score > BUY_THRESHOLD and bearish_score < 3.0:
                return "BUY"
            # In a strong uptrend, take profits if momentum wanes and sentiment turns
            if bearish_score > SELL_THRESHOLD and bullish_score < 3.0:
                return "SELL"
        elif long_ema_slope < 0: # Bearish trend
            if bearish_score > SELL_THRESHOLD and bullish_score < 3.0:
                return "SELL"
            # In a strong downtrend, consider buying only on extreme oversold + positive sentiment
            if bullish_score > BUY_THRESHOLD and bearish_score < 3.0 and net_sentiment_score > 1.5:
                 return "BUY"
    else: # Ranging / Choppy Market
        # In a ranging market, prioritize mean-reversion signals
        # Buy condition: Oversold signals + positive sentiment + weakening bearish momentum
        if (rsi < rsi_oversold or current_price < lower_band) and bullish_score > BUY_THRESHOLD and bearish_score < 3.5:
            return "BUY"
        # Sell condition: Overbought signals + negative sentiment + weakening bullish momentum
        if (rsi > rsi_overbought or current_price > upper_band) and bearish_score > SELL_THRESHOLD and bullish_score < 3.5:
            return "SELL"

    return "HOLD"