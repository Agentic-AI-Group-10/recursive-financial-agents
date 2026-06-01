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
    # Correctly initialize the EMA series
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
    
    # Correctly iterate over remaining deltas
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

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None

    # Ensure enough data for MACD line calculation
    if len(prices) < long_period:
        return None
        
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    # Align the series before subtraction
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return None
        
    signal_line_series = calculate_ema_series(macd_line, signal_period)
    
    if len(signal_line_series) == 0:
        return None
        
    histogram = macd_line[-1] - signal_line_series[-1]
    return histogram

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY:
    This version addresses the critical failure of passivity by introducing an
    asymmetric, faster exit logic to reduce drawdowns, while retaining the
    successful regime-switching and multi-factor entry confirmation.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Unchanged - Proven Robustness) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.0, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
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

    # --- 2. Technical Indicators & Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    MEDIUM_TERM_SMA_PERIOD = 50
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1, MEDIUM_TERM_SMA_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate all indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_histogram = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, macd_histogram, medium_sma]):
        return "HOLD"

    # Volatility Regime Detection (Unchanged - Proven Success)
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following (Unchanged - Proven Success) ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT_CEILING = 65
        RSI_OVERSOLD_FLOOR = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < RSI_OVERBOUGHT_CEILING:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > RSI_OVERSOLD_FLOOR:
            return "SELL"
    else:
        # === NORMAL MODE: Trend-following with Asymmetric, Faster Exit Logic ===
        # This new logic addresses the core failure of passivity and holding through drawdowns.
        
        # --- STRICT ENTRY CONDITIONS (BUY) ---
        # Requires strong confirmation from trend, momentum, and sentiment.
        is_bullish_trend = short_ema > long_ema
        is_bullish_momentum = macd_histogram > 0
        is_strong_rsi = rsi > 52  # Use >50 as a basic filter for bullish regime
        is_positive_sentiment = net_sentiment_score >= 1.0
        
        if is_bullish_trend and is_bullish_momentum and is_strong_rsi and is_positive_sentiment:
            return "BUY"

        # --- ASYMMETRIC EXIT CONDITIONS (SELL) ---
        # Faster, more sensitive exit to protect capital and lock in gains.
        # A position is sold if the medium-term trend is broken OR if momentum clearly weakens.
        price_breaks_medium_trend = current_price < medium_sma
        momentum_is_weakening = rsi < 45 # Exit if RSI shows loss of relative strength
        is_negative_sentiment = net_sentiment_score <= -1.0
        is_bearish_trend = short_ema < long_ema

        # The primary exit is breaking the medium-term SMA, acting as a stop-loss.
        # A secondary exit is a combination of negative trend and weakening momentum.
        if price_breaks_medium_trend or (is_bearish_trend and momentum_is_weakening) or (is_bearish_trend and is_negative_sentiment):
            return "SELL"

    return "HOLD"