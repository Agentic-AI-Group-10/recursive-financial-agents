import numpy as np
import re
import math

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
    try:
        import pandas as pd
        # Using pandas if available for a robust, standard implementation
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback to a manual calculation if pandas is not installed
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

def calculate_trend_consistency(prices, period=20):
    """Calculates a measure of trend consistency (lower is smoother)."""
    if len(prices) < period + 1:
        return None
    price_changes = np.diff(prices[-period-1:])
    if not np.any(price_changes):
        return 1.0
    std_dev_changes = np.std(price_changes)
    avg_abs_change = np.mean(np.abs(price_changes))
    if avg_abs_change == 0:
        return 1.0
    return std_dev_changes / avg_abs_change

def calculate_roc(prices, period=10):
    """Calculates the Rate of Change (ROC) to measure momentum velocity."""
    if len(prices) < period + 1:
        return None
    if prices[-1 - period] == 0:
        return 0.0
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_donchian_channels(prices, period=20):
    """Calculates the Donchian Channels for the latest price."""
    if len(prices) < period:
        return None, None
    prices_slice = prices[-period:]
    upper_channel = np.max(prices_slice)
    lower_channel = np.min(prices_slice)
    return upper_channel, lower_channel

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy featuring a structural bear market
    filter to prevent catastrophic drawdowns observed in past models.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Enhanced with market psychology keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "capitulation": 2.0,
        # Moderate Positive
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "disinflation": 2.0, "market rally": 2.0,
        # Mild Positive
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "easing tensions": 1.5, "consumer confidence": 1.5, "weak jobs report": 1.5, # Nuanced: good for rates
        # Strong Negative
        "panic": -3.5, "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "conflict": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        # Moderate Negative
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "sanctions": -2.5, "credit crunch": -2.5, "cpi beat": -2.5, "vix spike": -2.5,
        "supply chain disruption": -2.5, "geopolitical tensions": -2.5,
        # Mild Negative
        "hawkish": -2.0, "bearish": -2.0, "plunge": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "euphoria": -2.0, # Contrarian negative
        "tightening": -1.5, "miss estimates": -1.5, "downgrade": -1.5, "uncertainty": -1.5,
        "strong jobs report": -1.5, # Nuanced: bad for rates
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
    
    # Indicator Periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    SMA_200_PERIOD = 200 # **NEW** Structural regime filter
    RSI_PERIOD = 14
    BB_PERIOD = 20
    DONCHIAN_PERIOD = 20
    ROC_PERIOD = 10
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50
    TREND_CONSISTENCY_PERIOD = 20

    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1, SMA_200_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    sma_200 = calculate_sma(all_prices, SMA_200_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    prev_rsi = calculate_rsi(all_prices[:-1], RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    upper_donchian, lower_donchian = calculate_donchian_channels(all_prices, DONCHIAN_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    trend_consistency = calculate_trend_consistency(all_prices, TREND_CONSISTENCY_PERIOD)
    roc = calculate_roc(all_prices, ROC_PERIOD)

    if any(v is None for v in [short_ema, long_ema, sma_200, rsi, prev_rsi, upper_band, lower_band, upper_donchian, lower_donchian, short_atr, long_atr, trend_consistency, roc]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Define Market States / Regimes ---
    is_structural_bear_market = current_price < sma_200
    is_high_volatility = short_atr > (long_atr * 1.7)
    is_trending_market = trend_consistency < 1.2

    # --- 4. Hierarchical Decision Logic ---

    # === PRIMARY OVERRIDE: STRUCTURAL BEAR MARKET (CAPITAL PRESERVATION MODE) ===
    # Addresses the primary failure from past lessons: premature dip-buying in a sustained crash.
    if is_structural_bear_market:
        is_breaking_down = current_price < lower_donchian and roc < -1.0
        if is_breaking_down and net_sentiment_score < 1.0 and short_ema < long_ema:
            return "SELL"
        
        # High-risk contrarian BUYs are disabled by default in a bear market.
        # Only an exceptionally strong, multi-factor reversal signal would be considered.
        is_reversing_up = macd_histogram > prev_macd_histogram and macd_histogram < 0
        if rsi < 25 and is_reversing_up and net_sentiment_score > 4.5:
             return "BUY" # Rare, high-conviction capitulation buy.
        
        return "HOLD" # Default action in a bear market is to hold cash.

    # === NORMAL MARKET LOGIC (Price > SMA_200) ===
    if is_high_volatility:
        # Sub-Regime: High-Volatility Bull Market (e.g., sharp correction, V-recovery)
        BULLISH_SENTIMENT_THRESHOLD = 3.5
        BEARISH_SENTIMENT_THRESHOLD = -3.5
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and short_ema > long_ema and macd_histogram > 0 and rsi < 75:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and short_ema < long_ema and macd_histogram < 0 and rsi > 25:
            return "SELL"
    else:
        if is_trending_market:
            # Sub-Regime: Normal Trending Bull Market
            is_momentum_fading_up = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if (rsi > 78 or current_price > upper_band) and is_momentum_fading_up:
                return "SELL" # Take profit on signs of exhaustion

            is_pullback_buy = short_ema > long_ema and macd_histogram > 0 and rsi < 75 and roc > 0.1
            if is_pullback_buy and net_sentiment_score > -2.5:
                return "BUY"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            is_reversing_up = macd_histogram > prev_macd_histogram
            is_at_donchian_low = abs(current_price - lower_donchian) < (short_atr * 0.25)
            if rsi < 35 and current_price < lower_band and is_at_donchian_low and \
               net_sentiment_score > -3.0 and is_reversing_up and rsi > prev_rsi:
                return "BUY"
                
            is_reversing_down = macd_histogram < prev_macd_histogram
            is_at_donchian_high = abs(current_price - upper_donchian) < (short_atr * 0.25)
            if rsi > 65 and current_price > upper_band and is_at_donchian_high and \
               net_sentiment_score < 3.0 and is_reversing_down and rsi < prev_rsi:
                return "SELL"

    return "HOLD"