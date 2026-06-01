import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def _calculate_ema_series(prices, period):
    """
    Calculates a full series of Exponential Moving Averages (EMA).
    Internal helper function.
    """
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.zeros_like(prices_arr)
    # Calculate the initial SMA
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    # Calculate the rest of the EMAs
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    if len(deltas) < period:
        return None
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method
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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return upper_band, middle_band, lower_band

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that integrates MACD and Bollinger Bands
    for more responsive and confirmed signals within its adaptive framework.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Inherited and Refined) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat expectations": 1.5, "growth accelerates": 1.5, "recovery": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation fears": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss expectations": -1.5, "downgrade": -1.5, "tariff": -1.5,
        # Uncertainty/Neutral-Negative
        "uncertainty": -0.5, "volatility": -0.5, "mixed signals": -0.5
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
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_short_series = _calculate_ema_series(all_prices, SHORT_EMA_PERIOD)
    ema_long_series = _calculate_ema_series(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    upper_band, _, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)

    # Safeguard against None values from calculations
    if ema_short_series is None or ema_long_series is None or rsi is None or upper_band is None:
        return "HOLD"

    # Calculate MACD and Signal Line
    macd_line_series = ema_short_series - ema_long_series
    signal_line_series = _calculate_ema_series(macd_line_series, MACD_SIGNAL_PERIOD)
    if signal_line_series is None:
        return "HOLD"
        
    # Get latest values for decision making
    short_ema = ema_short_series[-1]
    long_ema = ema_long_series[-1]
    macd_line = macd_line_series[-1]
    macd_line_prev = macd_line_series[-2]
    signal_line = signal_line_series[-1]
    signal_line_prev = signal_line_series[-2]

    # Adaptive Volatility Regime (Inherited from parent)
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.6) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic with MACD & Bollinger Bands ---
    if is_high_volatility:
        # === CRISIS MODE: High-confluence, momentum-following, risk-off ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT = 60 # Be more conservative
        RSI_OVERSOLD = 40   # Be more conservative

        # BUY only with overwhelming evidence: underlying trend, momentum, and sentiment agree.
        if macd_line > signal_line and short_ema > long_ema and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and rsi < RSI_OVERBOUGHT:
            return "BUY"
        # SELL with strong evidence of breakdown.
        elif macd_line < signal_line and short_ema < long_ema and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and rsi > RSI_OVERSOLD:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.007 # Slightly wider threshold

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market (using MACD crossover for entry)
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            # Fresh bullish crossover confirms new upward momentum in an existing uptrend.
            is_bullish_crossover = macd_line > signal_line and macd_line_prev <= signal_line_prev
            if is_bullish_crossover and short_ema > long_ema and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
                return "BUY"
            
            # Fresh bearish crossover confirms new downward momentum in a downtrend.
            is_bearish_crossover = macd_line < signal_line and macd_line_prev >= signal_line_prev
            if is_bearish_crossover and short_ema < long_ema and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion with Bollinger Band confirmation)
            MEAN_REVERSION_RSI_OVERSOLD = 30
            MEAN_REVERSION_RSI_OVERBOUGHT = 70
            
            # Buy on extreme oversold conditions confirmed by price hitting the lower band.
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and current_price <= lower_band and net_sentiment_score > -2.0:
                return "BUY"
            # Sell on extreme overbought conditions confirmed by price hitting the upper band.
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and current_price >= upper_band and net_sentiment_score < 2.0:
                return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"