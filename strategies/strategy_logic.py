import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators (Self-Improved with Stochastic & EMA Slope) ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
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
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
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

def calculate_stochastic_oscillator(prices, k_period=14, d_period=3):
    """Calculates the Stochastic Oscillator (%K and %D) using only close prices."""
    if len(prices) < k_period:
        return None, None
    prices_arr = np.array(prices, dtype=float)
    
    # Calculate %K series
    percent_k_series = np.zeros(len(prices_arr) - k_period + 1)
    for i in range(len(percent_k_series)):
        window = prices_arr[i : i + k_period]
        high_k = np.max(window)
        low_k = np.min(window)
        if high_k == low_k:
            percent_k_series[i] = 50.0
        else:
            percent_k_series[i] = 100 * (prices_arr[i + k_period - 1] - low_k) / (high_k - low_k)
            
    if len(percent_k_series) < d_period:
        return percent_k_series[-1], None

    # Calculate %D (SMA of %K)
    percent_d_series = np.convolve(percent_k_series, np.ones(d_period), 'valid') / d_period
    
    return percent_k_series[-1], percent_d_series[-1]

def calculate_ema_slope(prices, period=50, lookback=5):
    """Calculates the normalized slope of an EMA line to gauge trend strength."""
    if len(prices) < period + lookback:
        return None
    ema_series = calculate_ema_series(prices, period)
    if len(ema_series) < lookback:
        return None
    recent_ema = ema_series[-lookback:]
    x = np.arange(lookback)
    y = recent_ema
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    avg_price = np.mean(prices[-period:])
    if avg_price == 0: return 0.0
    return (slope / avg_price) * 100

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using Stochastic Oscillator for
    mean-reversion confirmation and EMA slope for trend strength detection.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Expanded Lexicon) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "capitulation": 2.0, "supply chain easing": 2.0,
        # Moderate Positive
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "disinflation": 2.0, "market rally": 2.0, "vix crush": 2.0,
        # Mild Positive
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "easing tensions": 1.5, "consumer confidence": 1.5, "weak jobs report": 1.5, "de-escalation": 2.0,
        # Strong Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "conflict": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "geopolitical escalation": -3.0,
        # Moderate Negative
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "sanctions": -2.5, "credit crunch": -2.5, "cpi beat": -2.5, "euphoria": -2.0, "vix spike": -2.5,
        "supply chain disruption": -2.5,
        # Mild Negative
        "hawkish": -2.0, "bearish": -2.0, "plunge": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "tightening": -1.5, "miss estimates": -1.5,
        "downgrade": -1.5, "tariff": -1.5, "uncertainty": -1.5, "strong jobs report": -1.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Indicator Parameters
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    TREND_EMA_PERIOD = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50
    
    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1, TREND_EMA_PERIOD + 5, STOCH_K_PERIOD + STOCH_D_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    
    # **NEW** indicators for confirmation
    long_ema_slope = calculate_ema_slope(all_prices, TREND_EMA_PERIOD, lookback=5)
    stoch_k, stoch_d = calculate_stochastic_oscillator(all_prices, STOCH_K_PERIOD, STOCH_D_PERIOD)
    prev_stoch_k, prev_stoch_d = calculate_stochastic_oscillator(all_prices[:-1], STOCH_K_PERIOD, STOCH_D_PERIOD)

    # Null check for all required values
    if any(v is None for v in [short_ema, long_ema, rsi, upper_band, short_atr, long_atr, long_ema_slope, stoch_k, stoch_d, prev_stoch_k, prev_stoch_d]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]

    # Regime Detection
    is_high_volatility = short_atr > (long_atr * 1.65)
    is_trending_market = abs(long_ema_slope) > 0.05 # Trend is significant if slope > 0.05% per day

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, sentiment-driven trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 4.0
        BEARISH_SENTIMENT_THRESHOLD = -4.0
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and short_ema > long_ema and macd_histogram > 0 and rsi < 80:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and short_ema < long_ema and macd_histogram < 0 and rsi > 20:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive with Enhanced Confirmation Logic ===
        if is_trending_market:
            # Sub-Regime: Normal Trending Market
            bullish_trend = short_ema > long_ema and long_ema_slope > 0.05
            bearish_trend = short_ema < long_ema and long_ema_slope < -0.05
            
            # Exit over-extended positions
            if bullish_trend and rsi > 78:
                return "SELL"
            if bearish_trend and rsi < 22:
                return "BUY"

            # Entry requires trend confirmation from EMA slope
            if bullish_trend and macd_histogram > 0 and rsi < 75 and net_sentiment_score > -2.0:
                return "BUY"
            
            if bearish_trend and macd_histogram < 0 and rsi > 25 and net_sentiment_score < 2.0:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            
            # **IMPROVEMENT**: Stricter entry requires Stochastic crossover confirmation.
            is_stoch_bullish_crossover = prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d
            if rsi < 32 and current_price < lower_band and is_stoch_bullish_crossover and prev_stoch_k < 25 and net_sentiment_score > -3.5:
                return "BUY"
                
            is_stoch_bearish_crossover = prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d
            if rsi > 68 and current_price > upper_band and is_stoch_bearish_crossover and prev_stoch_k > 75 and net_sentiment_score < 3.5:
                return "SELL"

    return "HOLD"