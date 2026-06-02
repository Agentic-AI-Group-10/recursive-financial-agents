import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
    ema_values[0] = np.mean(data_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(1, len(ema_values)):
        ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) using Wilder's smoothing method.
    """
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period + signal_period:
        return None, None, None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return None, None, None
        
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
    """
    Calculates the Average True Range (ATR) using absolute price changes as a proxy for True Range,
    as only closing prices are available.
    """
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    # Using absolute price change as a proxy for True Range
    true_ranges = np.abs(np.diff(prices_arr))
    
    # Wilder's smoothing for ATR
    atr = np.mean(true_ranges[:period])
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
        
    return atr

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy with proactive exit logic
    and an ATR-based volatility filter for enhanced robustness.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (with minor keyword enhancements) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.0, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5, "better than expected": 1.5, "strong guidance": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0,
        "worse than expected": -1.5, "weak guidance": -1.5, "geopolitical tensions": -2.0
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
    
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 14 # Aligned with ATR standard
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    
    # **IMPROVEMENT**: ATR-based volatility regime detection
    short_atr = calculate_atr(all_prices, VOL_SHORT_PERIOD)
    long_atr = calculate_atr(all_prices, VOL_LONG_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, upper_band, short_atr, long_atr]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Regime is high volatility if recent ATR is >160% of long-term ATR and is significant relative to price
    is_high_volatility = (short_atr > long_atr * 1.6) and (short_atr / current_price > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following (Unchanged from successful parent) ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < 65:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > 35:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive with Proactive Exit Logic ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            
            # **IMPROVEMENT**: Stricter proactive profit-taking on trend exhaustion
            is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if bullish_trend and rsi > 75 and is_momentum_fading and current_price > upper_band:
                return "SELL"

            # Relaxed entry, using sentiment as a veto
            if bullish_trend and macd_histogram > 0 and rsi < 75 and net_sentiment_score > -1.5:
                return "BUY"
            
            # Standard sell signal for bearish trend
            if bearish_trend and macd_histogram < 0 and rsi > 25 and net_sentiment_score < 1.5:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion)
            MEDIUM_TERM_SMA_PERIOD = 50
            if len(all_prices) < MEDIUM_TERM_SMA_PERIOD:
                return "HOLD"
            
            medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)
            if medium_sma is None:
                return "HOLD"

            # **IMPROVEMENT**: More flexible mean-reversion entry
            is_momentum_turning_positive = macd_histogram > 0 and prev_macd_histogram <= 0
            buy_the_dip_signal = (rsi < 35 and current_price < lower_band)
            
            # Buy if oversold and either medium trend is up OR momentum is just turning positive
            if buy_the_dip_signal and net_sentiment_score > -2.0 and (current_price > medium_sma or is_momentum_turning_positive):
                return "BUY"
            
            # Sell the rip if overbought (unchanged from parent)
            elif (rsi > 70 and current_price > upper_band) and (net_sentiment_score < 2.0):
                return "SELL"

    return "HOLD"