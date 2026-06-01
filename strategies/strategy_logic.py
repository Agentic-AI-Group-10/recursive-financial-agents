import numpy as np
import re
from collections import defaultdict

# --- Helper Functions for Technical Indicators ---

def _calculate_ema_series(prices, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(prices) < period:
        return np.array([])
    prices_arr = np.array(prices, dtype=float)
    ema_series = np.full(len(prices_arr), np.nan)
    ema_series[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_series[i] = (prices_arr[i] - ema_series[i-1]) * multiplier + ema_series[i-1]
    return ema_series

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

    # Use Wilder's smoothing method (equivalent to a specific EMA)
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

def calculate_macd(prices, short_period, long_period, signal_period):
    """Calculates MACD line and Signal line."""
    if len(prices) < long_period + signal_period:
        return None, None
    
    short_ema_series = _calculate_ema_series(prices, short_period)
    long_ema_series = _calculate_ema_series(prices, long_period)
    
    if len(long_ema_series) == 0:
        return None, None

    macd_line_series = short_ema_series[long_period-1:] - long_ema_series[long_period-1:]
    
    if len(macd_line_series) < signal_period:
        return None, None
        
    signal_line_series = _calculate_ema_series(macd_line_series, signal_period)

    if len(signal_line_series) == 0 or np.isnan(signal_line_series[-1]):
        return None, None
        
    return macd_line_series[-1], signal_line_series[-1]

def calculate_bollinger_bands(prices, period, num_std_dev):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    relevant_prices = prices[-period:]
    middle_band = np.mean(relevant_prices)
    std_dev = np.std(relevant_prices)
    
    upper_band = middle_band + (num_std_dev * std_dev)
    lower_band = middle_band - (num_std_dev * std_dev)
    
    return middle_band, upper_band, lower_band

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses MACD for trend confirmation
    and Bollinger Bands for dynamic mean-reversion signals.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Capped Keyword Influence ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai breakthrough": 1.5, "beat estimates": 1.5, "growth": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation fears": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical tension": -1.5, "miss estimates": -1.5, "downgrade": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    keyword_counts = defaultdict(int)
    MAX_KEYWORD_HITS = 2

    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            if keyword_counts[keyword] < MAX_KEYWORD_HITS:
                pre_context = context_lower[max(0, match.start() - 30):match.start()]
                is_negated = any(neg_word in pre_context for neg_word in negation_words)
                net_sentiment_score += -weight if is_negated else weight
                keyword_counts[keyword] += 1

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BBAND_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + SIGNAL_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1, BBAND_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = _calculate_ema_series(all_prices, SHORT_EMA_PERIOD)[-1]
    long_ema = _calculate_ema_series(all_prices, LONG_EMA_PERIOD)[-1]
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)
    _, upper_bband, lower_bband = calculate_bollinger_bands(all_prices, BBAND_PERIOD, 2.0)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line, upper_bband, lower_bband]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, MACD-confirmed trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        
        is_bullish_trend = short_ema > long_ema and macd_line > signal_line
        is_bearish_trend = short_ema < long_ema and macd_line < signal_line
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < 70:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > 30:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.0075

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market (MACD Confirmed)
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            is_bullish_trend = short_ema > long_ema and macd_line > signal_line
            is_bearish_trend = short_ema < long_ema and macd_line < signal_line

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < 75:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > 25:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Bollinger Band Mean-Reversion)
            if current_price < lower_bband and rsi < 30 and net_sentiment_score > -2.0:
                return "BUY"
            elif current_price > upper_bband and rsi > 70 and net_sentiment_score < 2.0:
                return "SELL"

    return "HOLD"