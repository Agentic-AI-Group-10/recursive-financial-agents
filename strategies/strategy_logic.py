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
    # Using pandas for a robust and standard EMA calculation
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback to manual calculation if pandas is not available
        ema_values = np.zeros_like(data_arr)
        ema_values[0] = data_arr[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = deltas * (deltas > 0)
    losses = -deltas * (deltas < 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None

    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    
    macd_line = short_ema_full - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    
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

def calculate_roc(prices, period):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy with enhanced signal conviction
    using momentum confirmation and more robust choppy market detection.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (IMPROVED KEYWORDS) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "stimulus": 2.0,
        "soft landing": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0,
        "surge": 2.0, "strong earnings": 2.0, "cooling inflation": 2.0, "disinflation": 2.0,
        "ai boom": 2.5, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5, "supply chain easing": 1.5,
        "recession": -3.0, "crisis": -3.0, "default": -3.0, "rate hike": -2.5,
        "bankruptcy": -2.5, "hard landing": -2.5, "stagflation": -2.5,
        "quantitative tightening": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0
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
    ROC_PERIOD = 3
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    roc = calculate_roc(all_prices, ROC_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, upper_band, roc]) or macd_hist_series is None or len(macd_hist_series) < 15:
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
        # === CRISIS MODE: High-conviction trend-following (Unchanged) ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < 65:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > 35:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive with Enhanced Conviction Filters ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        
        # **IMPROVEMENT**: More robust choppy market detection using MACD zero crossings
        recent_hist = macd_hist_series[-10:]
        macd_zero_crossings = np.sum(np.diff(np.sign(recent_hist)) != 0)
        is_choppy_market = trend_strength < 0.007 or macd_zero_crossings > 3

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            
            # Proactive profit-taking / trend exhaustion signal (Unchanged)
            is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if bullish_trend and rsi > 78 and is_momentum_fading:
                return "SELL"

            # **IMPROVEMENT**: Entry signals now require momentum confirmation via ROC
            ROC_CONFIRMATION_THRESHOLD = 0.5 # e.g., 0.5% move in the last 3 days
            if bullish_trend and macd_histogram > 0 and roc > ROC_CONFIRMATION_THRESHOLD and rsi < 75 and net_sentiment_score > -1.5:
                return "BUY"
            
            if bearish_trend and macd_histogram < 0 and roc < -ROC_CONFIRMATION_THRESHOLD and rsi > 25 and net_sentiment_score < 1.5:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion, unchanged)
            MEDIUM_TERM_SMA_PERIOD = 50
            if len(all_prices) < MEDIUM_TERM_SMA_PERIOD:
                return "HOLD"
            
            medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)
            if medium_sma is None:
                return "HOLD"

            # Buy the dip if confirmed by RSI, Bollinger Bands, and medium-term trend
            if (rsi < 30 and current_price < lower_band) and \
               (net_sentiment_score > -2.0) and (current_price > medium_sma):
                return "BUY"
            # Sell the rip if confirmed by RSI and Bollinger Bands
            elif (rsi > 70 and current_price > upper_band) and \
                 (net_sentiment_score < 2.0):
                return "SELL"

    return "HOLD"