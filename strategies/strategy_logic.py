import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.zeros_like(prices_arr, dtype=float)
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values[-1]

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

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculates MACD, Signal Line, and Histogram for the latest price.
    Returns (macd_line, signal_line, macd_histogram)
    """
    # Need enough data for the long EMA plus the signal line EMA
    if len(prices) < long_period + signal_period:
        return None, None, None

    prices_arr = np.array(prices, dtype=float)

    # Calculate EMAs for the entire series to get accurate recent values
    # Short EMA
    ema_short_values = np.zeros_like(prices_arr, dtype=float)
    multiplier_short = 2 / (short_period + 1)
    ema_short_values[short_period - 1] = np.mean(prices_arr[:short_period])
    for i in range(short_period, len(prices_arr)):
        ema_short_values[i] = (prices_arr[i] - ema_short_values[i-1]) * multiplier_short + ema_short_values[i-1]

    # Long EMA
    ema_long_values = np.zeros_like(prices_arr, dtype=float)
    multiplier_long = 2 / (long_period + 1)
    ema_long_values[long_period - 1] = np.mean(prices_arr[:long_period])
    for i in range(long_period, len(prices_arr)):
        ema_long_values[i] = (prices_arr[i] - ema_long_values[i-1]) * multiplier_long + ema_long_values[i-1]

    macd_line_values = ema_short_values - ema_long_values

    # Signal Line (EMA of MACD line)
    signal_line_values = np.zeros_like(macd_line_values, dtype=float)
    multiplier_signal = 2 / (signal_period + 1)
    valid_macd_start_index = long_period - 1
    signal_line_values[valid_macd_start_index + signal_period - 1] = np.mean(macd_line_values[valid_macd_start_index : valid_macd_start_index + signal_period])
    for i in range(valid_macd_start_index + signal_period, len(prices_arr)):
        signal_line_values[i] = (macd_line_values[i] - signal_line_values[i-1]) * multiplier_signal + signal_line_values[i-1]

    macd_histogram_values = macd_line_values - signal_line_values

    return macd_line_values[-1], signal_line_values[-1], macd_histogram_values[-1]

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy that addresses passivity by decoupling exit logic from
    entry logic and incorporates MACD for better momentum analysis.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Inherited from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Data Preparation ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)

    # Safeguard against None values
    if any(v is None for v in [short_ema, long_ema, rsi, macd_hist]):
        return "HOLD"

    # --- 3. Defensive SELL Logic (Capital Preservation) ---
    # This logic is checked first to exit positions quickly if conditions sour.
    # It's more sensitive than the logic for initiating a new short position.
    is_bearish_trend_cross = short_ema < long_ema
    is_momentum_collapsing = macd_hist < -0.05 * (current_price / 100) # Dynamic threshold
    is_catastrophic_news = net_sentiment_score <= -3.0

    if is_bearish_trend_cross or is_momentum_collapsing or is_catastrophic_news:
        return "SELL"

    # --- 4. Regime Detection & Offensive BUY/SELL Logic ---
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    is_bullish_trend = short_ema > long_ema
    is_bullish_momentum = macd_hist > 0

    if is_high_volatility:
        # === CRISIS MODE: High-conviction entries only ===
        if is_bullish_trend and is_bullish_momentum and net_sentiment_score >= 2.0 and rsi < 65:
            return "BUY"
    else:
        # === NORMAL MODE: Adaptive strategy ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            if is_bullish_trend and is_bullish_momentum and net_sentiment_score >= 1.0 and rsi < 70:
                return "BUY"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion)
            if rsi < 25 and net_sentiment_score > -1.5:
                return "BUY"
            # In choppy markets, we use the primary defensive SELL for exits, but won't initiate new shorts.
            # We can add a specific mean-reversion sell for profit-taking if desired.
            elif rsi > 75 and net_sentiment_score < 1.5:
                return "SELL"

    # Default action is to hold.
    return "HOLD"