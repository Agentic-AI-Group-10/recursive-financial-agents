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

def calculate_adx(prices, period=14):
    """
    Calculates the Average Directional Index (ADX).
    Since only close prices are available, price delta is used as a proxy for True Range.
    """
    if len(prices) < 2 * period + 1:
        return None

    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    plus_dm = np.where(deltas > 0, deltas, 0)
    minus_dm = np.where(deltas < 0, -deltas, 0)
    
    # Use price delta as a proxy for True Range
    true_range = np.abs(deltas)

    def wilder_smooth(data, p):
        smoothed = np.zeros(len(data) - p + 1)
        if len(smoothed) == 0: return np.array([])
        smoothed[0] = np.mean(data[:p])
        for i in range(1, len(smoothed)):
            smoothed[i] = (smoothed[i-1] * (p - 1) + data[i + p - 1]) / p
        return smoothed

    atr_series = wilder_smooth(true_range, period)
    plus_dm_series = wilder_smooth(plus_dm, period)
    minus_dm_series = wilder_smooth(minus_dm, period)

    if len(atr_series) == 0: return None
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * (plus_dm_series / atr_series)
        minus_di = 100 * (minus_dm_series / atr_series)
        plus_di[atr_series == 0] = 0
        minus_di[atr_series == 0] = 0

    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx_denominator = plus_di + minus_di
        dx_series = 100 * (np.abs(plus_di - minus_di) / dx_denominator)
        dx_series[dx_denominator == 0] = 0
    
    if len(dx_series) < period:
        return None
    
    adx = wilder_smooth(dx_series, period)
    return adx[-1] if len(adx) > 0 else None

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using ADX for robust
    trend identification and an "indecisive" zone to reduce noise.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Refined) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.0, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5, "better than expected": 1.5, "jobless claims fall": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0,
        "worse than expected": -1.5, "jobless claims rise": -1.5, "default risk": -2.5
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
    ADX_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1, 2 * ADX_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)

    if any(v is None for v in [short_ema, long_ema, rsi, adx, upper_band]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices[1:]) / np.array(all_prices[:-1]))
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        if short_ema > long_ema and macd_histogram > 0 and rsi < 65 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
            return "BUY"
        elif short_ema < long_ema and macd_histogram < 0 and rsi > 35 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "SELL"
    else:
        # === NORMAL MODE: ADX-based regime switching ===
        if adx > 25:
            # Sub-Regime: Trending Market
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            
            # Proactive profit-taking / trend exhaustion signal
            is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if bullish_trend and rsi > 78 and is_momentum_fading:
                return "SELL"

            # Buy on confirmed bullish trend, with sentiment as a soft veto
            if bullish_trend and macd_histogram > 0 and rsi < 75 and net_sentiment_score > -1.5:
                return "BUY"
            
            # Sell on confirmed bearish trend, with sentiment as a soft veto
            if bearish_trend and macd_histogram < 0 and rsi > 25 and net_sentiment_score < 1.5:
                return "SELL"
        elif adx < 20:
            # Sub-Regime: Ranging / Choppy Market (Mean-Reversion)
            MEDIUM_TERM_SMA_PERIOD = 50
            if len(all_prices) < MEDIUM_TERM_SMA_PERIOD:
                return "HOLD"
            
            medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)
            if medium_sma is None: return "HOLD"

            # Buy the dip if confirmed by RSI, BBands, sentiment, and long-term trend
            if (rsi < 30 and current_price < lower_band) and \
               (net_sentiment_score > -2.0) and (current_price > medium_sma):
                return "BUY"
            # Sell the rip if confirmed by RSI and BBands
            elif (rsi > 70 and current_price > upper_band) and \
                 (net_sentiment_score < 2.0):
                return "SELL"
        else:
            # Sub-Regime: Indecisive Market (ADX between 20 and 25)
            # Avoid trading in a directionless market to reduce noise.
            return "HOLD"

    return "HOLD"