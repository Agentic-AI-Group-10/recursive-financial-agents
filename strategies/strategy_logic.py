import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

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

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

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

def calculate_vamo(prices, period=14):
    """
    Calculates the Volatility-Adjusted Momentum Oscillator (VAMO).
    This measures the N-period price change in units of N-period ATR.
    A value of +2.0 means the price has moved up by 2x the average daily range.
    """
    if len(prices) < period + 1:
        return None
    atr = calculate_atr(prices, period)
    if atr is None or atr == 0:
        return 0.0
    price_change = prices[-1] - prices[-period]
    return price_change / atr

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy using a unified conviction score and a robust "Phoenix"
    re-entry signal for crash scenarios, addressing past failures in volatile markets.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive (Macro)
        "fed pivot": 4.0, "quantitative easing": 3.5, "stimulus": 3.0, "rate cut": 3.0,
        "soft landing": 2.5, "cooling inflation": 2.5, "disinflation": 2.0,
        # Strong Positive (Market)
        "ai boom": 2.5, "record high": 2.0, "strong earnings": 2.0, "breakthrough": 2.0,
        # Mild Positive
        "dovish": 1.5, "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.0,
        # Contrarian Bullish (Use with caution, lower weight)
        "capitulation": 1.5, "panic selling": 1.0, "extreme fear": 1.0,
        # Strong Negative (Macro)
        "black swan": -4.0, "systemic risk": -4.0, "crisis": -3.5, "recession": -3.0,
        "stagflation": -3.0, "yield curve inversion": -3.5, "hot inflation": -3.0,
        "quantitative tightening": -3.0, "war": -3.0,
        # Strong Negative (Market)
        "credit crunch": -3.0, "contagion": -3.0, "vix spike": -2.5, "hard landing": -2.5,
        # Mild Negative
        "rate hike": -2.0, "hawkish": -2.0, "bearish": -1.5, "sell-off": -1.5,
        "market turmoil": -1.5, "uncertainty": -1.0,
        # Contrarian Bearish (Greed)
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Data Preparation ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    TREND_SMA_20 = 20
    TREND_SMA_50 = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50
    VAMO_PERIOD = 14

    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1, TREND_SMA_50 + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    sma_20 = calculate_sma(all_prices, TREND_SMA_20)
    sma_50 = calculate_sma(all_prices, TREND_SMA_50)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    vamo = calculate_vamo(all_prices, VAMO_PERIOD)
    
    # Previous day's values for crossover detection
    prev_prices = all_prices[:-1]
    prev_sma_20 = calculate_sma(prev_prices, TREND_SMA_20)

    if any(v is None for v in [short_ema, long_ema, sma_20, sma_50, rsi, upper_band, short_atr, long_atr, vamo, prev_sma_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_long_term_bearish = current_price < sma_50
    is_crash_environment = is_long_term_bearish and is_high_volatility

    # --- 4. Decision Logic ---

    # === REGIME 1: CRASH ENVIRONMENT ===
    # Priority is capital preservation. Avoid buying unless a high-conviction "Phoenix" reversal occurs.
    if is_crash_environment:
        # "Phoenix" Re-entry Signal (High-conviction BUY)
        is_reclaiming_trend = current_price > sma_20 and prev_prices[-1] < prev_sma_20
        is_momentum_confirmed = macd_histogram > 0
        has_macro_support = net_sentiment_score > 3.5 # Requires strong positive news (stimulus, QE, etc.)
        
        if is_reclaiming_trend and is_momentum_confirmed and has_macro_support:
            return "BUY"

        # Default action in a crash: stay out of the market.
        if current_price < sma_20:
            return "SELL"
        
        return "HOLD"

    # === REGIME 2: NORMAL ENVIRONMENT ===
    # Use a unified conviction score to make decisions.
    buy_score = 0.0
    sell_score = 0.0

    # --- Scoring Components ---
    # 1. Trend Score (Weight: 2.0)
    if current_price > short_ema and short_ema > long_ema and current_price > sma_20:
        buy_score += 2.0
    elif current_price < short_ema and short_ema < long_ema and current_price < sma_20:
        sell_score += 2.0

    # 2. Momentum Score (Weight: 2.5)
    if vamo > 1.25 and rsi > 52: # Strong upward momentum relative to volatility
        buy_score += 1.5
    if vamo < -1.25 and rsi < 48: # Strong downward momentum relative to volatility
        sell_score += 1.5
    if macd_histogram > 0 and macd_histogram > prev_macd_histogram: # Accelerating positive momentum
        buy_score += 1.0
    if macd_histogram < 0 and macd_histogram < prev_macd_histogram: # Accelerating negative momentum
        sell_score += 1.0

    # 3. Sentiment Score (Weight: Scaled)
    buy_score += max(0, net_sentiment_score / 2.0)
    sell_score += max(0, -net_sentiment_score / 2.0)

    # 4. Overbought/Oversold Mean Reversion (Can act as entry or exit signal)
    if rsi > 78 and current_price > upper_band:
        sell_score += 1.5 # Profit-taking or short signal
    if rsi < 22 and current_price < lower_band:
        buy_score += 1.5 # Contrarian buy signal

    # --- Final Decision ---
    CONVICTION_THRESHOLD = 4.0
    if buy_score >= CONVICTION_THRESHOLD and sell_score < (buy_score * 0.5):
        return "BUY"
    
    if sell_score >= CONVICTION_THRESHOLD and buy_score < (sell_score * 0.5):
        return "SELL"

    return "HOLD"